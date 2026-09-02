//! Vector storage and exact similarity search.
//!
//! SQLite vec0 remains the durable vector store and rollback path. The default
//! search mode also maintains a flat, little-endian `f32` sidecar in the index
//! directory and scans it with SimSIMD. Set `VERA_VECTOR_SCAN=vec0` to select
//! the SQLite KNN path for comparison or rollback.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, atomic::AtomicU64};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use memmap2::Mmap;
use rusqlite::{Connection, OptionalExtension, ffi::sqlite3_auto_extension, params};
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use sqlite_vec::sqlite3_vec_init;
use zerocopy::IntoBytes;

const PREFIX_RANGE_SQL: &str =
    "SELECT rowid FROM chunk_id_map WHERE chunk_id >= ?1 AND chunk_id < ?2";
const PREFIX_LOWER_BOUND_SQL: &str = "SELECT rowid FROM chunk_id_map WHERE chunk_id >= ?1";
const FLAT_FILE_NAME: &str = "vectors.f32";
const TOMBSTONE_FILE_NAME: &str = "vectors.tombs";
const MANIFEST_FILE_NAME: &str = "vectors.manifest";
const FLAT_MANIFEST_VERSION: u32 = 2;
const VECTOR_SCAN_ENV: &str = "VERA_VECTOR_SCAN";
const DATABASE_ID_KEY: &str = "database_id";

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorScanMode {
    Flat,
    Vec0,
}

impl VectorScanMode {
    fn from_env() -> Self {
        match std::env::var(VECTOR_SCAN_ENV) {
            Ok(value) if value.eq_ignore_ascii_case("vec0") => Self::Vec0,
            Ok(value) if value.eq_ignore_ascii_case("flat") || value.trim().is_empty() => {
                Self::Flat
            }
            Ok(value) => {
                tracing::warn!(
                    value = %value,
                    default = "flat",
                    "unknown vector scan mode; using flat scan"
                );
                Self::Flat
            }
            Err(_) => Self::Flat,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlatManifest {
    version: u32,
    database_id: String,
    dim: usize,
    max_rowid: i64,
    generation: u64,
    tombstone_count: u64,
    flat_bytes: u64,
}

enum FlatData {
    Empty,
    Memory(Vec<f32>),
    Mmap(Mmap),
}

impl FlatData {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Empty => &[],
            Self::Memory(values) => values,
            Self::Mmap(mmap) => {
                // The flat file is written as little-endian f32 and only mapped
                // directly on little-endian targets, which is the format used
                // by Vera's supported native builds.
                debug_assert_eq!(mmap.len() % std::mem::size_of::<f32>(), 0);
                // SAFETY: the mmap starts at a page-aligned address and its
                // length is validated as a multiple of four. Updates preserve
                // existing row offsets, and callers reload after an append or
                // atomic sidecar replacement changes the mapped file size.
                unsafe {
                    std::slice::from_raw_parts(
                        mmap.as_ptr().cast::<f32>(),
                        mmap.len() / std::mem::size_of::<f32>(),
                    )
                }
            }
        }
    }

    fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}

struct FlatSnapshot {
    manifest: FlatManifest,
    data: FlatData,
    tombstones: Vec<u8>,
    manifest_mtime: Option<SystemTime>,
}

struct DiskFlatStorage {
    dim: usize,
    flat_path: PathBuf,
    tombstone_path: PathBuf,
    manifest_path: PathBuf,
    snapshot: FlatSnapshot,
}

struct FlatPaths<'a> {
    flat: &'a Path,
    tombstone: &'a Path,
    manifest: &'a Path,
}

struct MemoryFlatStorage {
    dim: usize,
    snapshot: FlatSnapshot,
}

enum FlatStorage {
    Disk(DiskFlatStorage),
    Memory(MemoryFlatStorage),
}

#[derive(Debug, Clone, Copy)]
struct DistanceCandidate {
    rowid: i64,
    distance: f64,
}

impl PartialEq for DistanceCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == Ordering::Equal && self.rowid == other.rowid
    }
}

impl Eq for DistanceCandidate {}

impl PartialOrd for DistanceCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.rowid.cmp(&other.rowid))
    }
}

impl FlatSnapshot {
    fn empty(dim: usize, generation: u64) -> Self {
        Self {
            manifest: FlatManifest {
                version: FLAT_MANIFEST_VERSION,
                database_id: String::new(),
                dim,
                max_rowid: 0,
                generation,
                tombstone_count: 0,
                flat_bytes: 0,
            },
            data: FlatData::Empty,
            tombstones: Vec::new(),
            manifest_mtime: None,
        }
    }
}

impl FlatStorage {
    fn open_disk(conn: &Connection, dim: usize, sidecar_dir: &Path) -> Result<Self> {
        let storage = DiskFlatStorage {
            dim,
            flat_path: sidecar_dir.join(FLAT_FILE_NAME),
            tombstone_path: sidecar_dir.join(TOMBSTONE_FILE_NAME),
            manifest_path: sidecar_dir.join(MANIFEST_FILE_NAME),
            snapshot: FlatSnapshot::empty(dim, 0),
        };
        let mut storage = Self::Disk(storage);
        storage.reload_disk(conn)?;
        Ok(storage)
    }

    fn refresh(&mut self, conn: &Connection) -> Result<()> {
        let Self::Disk(storage) = self else {
            return Ok(());
        };
        let generation = current_generation(conn)?;
        let database_id = current_database_id(conn)?;
        let mtime = manifest_mtime(&storage.manifest_path);
        if storage.snapshot.manifest.database_id != database_id
            || storage.snapshot.manifest.generation != generation
            || storage.snapshot.manifest_mtime != mtime
        {
            self.reload_disk(conn)?;
        }
        Ok(())
    }

    fn reload_disk(&mut self, conn: &Connection) -> Result<()> {
        let Self::Disk(storage) = self else {
            return Ok(());
        };
        let snapshot = load_or_rebuild_disk_snapshot(
            conn,
            storage.dim,
            FlatPaths {
                flat: &storage.flat_path,
                tombstone: &storage.tombstone_path,
                manifest: &storage.manifest_path,
            },
        )?;
        storage.snapshot = snapshot;
        Ok(())
    }

    fn apply_update(
        &mut self,
        conn: &Connection,
        inserts: &[(i64, &[f32])],
        tombstone_rowids: &[i64],
        generation: u64,
    ) -> Result<()> {
        match self {
            Self::Memory(storage) => storage.apply_update(inserts, tombstone_rowids, generation),
            Self::Disk(storage) => {
                apply_disk_update(conn, storage, inserts, tombstone_rowids, generation)?;
                self.reload_disk(conn)
            }
        }
    }

    fn search(
        &mut self,
        conn: &Connection,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<DistanceCandidate>> {
        self.refresh(conn)?;
        let snapshot = match self {
            Self::Disk(storage) => &storage.snapshot,
            Self::Memory(storage) => &storage.snapshot,
        };
        scan_snapshot(snapshot, query, limit)
    }

    fn search_filtered(
        &mut self,
        conn: &Connection,
        query: &[f32],
        limit: usize,
        map: &crate::storage::eligibility::EligibilityMap,
        query_elig: &crate::storage::eligibility::QueryEligibility,
    ) -> Result<Vec<DistanceCandidate>> {
        self.refresh(conn)?;
        let snapshot = match self {
            Self::Disk(storage) => &storage.snapshot,
            Self::Memory(storage) => &storage.snapshot,
        };
        scan_snapshot_filtered(snapshot, query, limit, map, query_elig)
    }
}

impl MemoryFlatStorage {
    fn apply_update(
        &mut self,
        inserts: &[(i64, &[f32])],
        tombstone_rowids: &[i64],
        generation: u64,
    ) -> Result<()> {
        let max_rowid = inserts
            .iter()
            .map(|(rowid, _)| *rowid)
            .chain(tombstone_rowids.iter().copied())
            .chain(std::iter::once(self.snapshot.manifest.max_rowid))
            .max()
            .unwrap_or(0);
        if max_rowid < 0 {
            anyhow::bail!("negative vector rowid")
        }
        let max_rowid = max_rowid as usize;
        let values_len = max_rowid
            .checked_mul(self.dim)
            .context("flat vector storage size overflow")?;
        let mut values = self.snapshot.data.to_vec();
        values.resize(values_len, 0.0);
        let mut tombstones = self.snapshot.tombstones.clone();
        resize_tombstones(
            &mut tombstones,
            self.snapshot.manifest.max_rowid,
            max_rowid as i64,
        )?;

        for rowid in tombstone_rowids {
            set_tombstone(&mut tombstones, *rowid, true)?;
        }
        for (rowid, vector) in inserts {
            if vector.len() != self.dim {
                anyhow::bail!(
                    "vector dimension mismatch: expected {}, got {}",
                    self.dim,
                    vector.len()
                );
            }
            let rowid = usize::try_from(*rowid).context("vector rowid out of range")?;
            if rowid == 0 {
                anyhow::bail!("vector rowid must be positive")
            }
            let start = (rowid - 1)
                .checked_mul(self.dim)
                .context("flat vector storage offset overflow")?;
            let end = start + self.dim;
            values[start..end].copy_from_slice(vector);
            set_tombstone(&mut tombstones, rowid as i64, false)?;
        }

        let tombstone_count = count_tombstones(&tombstones, max_rowid as i64);
        self.snapshot = FlatSnapshot {
            manifest: FlatManifest {
                version: FLAT_MANIFEST_VERSION,
                database_id: String::new(),
                dim: self.dim,
                max_rowid: max_rowid as i64,
                generation,
                tombstone_count,
                flat_bytes: (values_len * std::mem::size_of::<f32>()) as u64,
            },
            data: if values.is_empty() {
                FlatData::Empty
            } else {
                FlatData::Memory(values)
            },
            tombstones,
            manifest_mtime: None,
        };
        Ok(())
    }
}

fn scan_snapshot(
    snapshot: &FlatSnapshot,
    query: &[f32],
    limit: usize,
) -> Result<Vec<DistanceCandidate>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let dim = snapshot.manifest.dim;
    if dim == 0 || query.len() != dim {
        anyhow::bail!(
            "flat vector snapshot dimension mismatch: expected {}, got {}",
            dim,
            query.len()
        );
    }
    // Flat scan is bounded by the actual vector count, not the sqlite-vec KNN cap.
    let values_len = snapshot.data.as_slice().len();
    let available = values_len / dim;
    let max_k = limit.min(available);
    if max_k == 0 {
        return Ok(Vec::new());
    }
    let values = snapshot.data.as_slice();
    let mut heap = BinaryHeap::with_capacity(max_k);
    for (index, vector) in values.chunks_exact(dim).enumerate() {
        let rowid = index as i64 + 1;
        if is_tombstoned(&snapshot.tombstones, rowid) {
            continue;
        }
        let distance = f32::euclidean(query, vector)
            .context("SimSIMD returned no distance for equal vector dimensions")?;
        let candidate = DistanceCandidate { rowid, distance };
        if heap.len() < max_k {
            heap.push(candidate);
        } else if candidate
            < *heap
                .peek()
                .context("flat top-k heap is unexpectedly empty")?
        {
            heap.pop();
            heap.push(candidate);
        }
    }

    let mut results: Vec<_> = heap.into_vec();
    results.sort();
    Ok(results)
}

fn scan_snapshot_filtered(
    snapshot: &FlatSnapshot,
    query: &[f32],
    limit: usize,
    map: &crate::storage::eligibility::EligibilityMap,
    query_elig: &crate::storage::eligibility::QueryEligibility,
) -> Result<Vec<DistanceCandidate>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    if query_elig.is_empty() {
        return Ok(Vec::new());
    }
    let dim = snapshot.manifest.dim;
    if dim == 0 || query.len() != dim {
        anyhow::bail!(
            "flat vector snapshot dimension mismatch: expected {}, got {}",
            dim,
            query.len()
        );
    }
    let values_len = snapshot.data.as_slice().len();
    let available = values_len / dim;
    let map_len = map.path_ids.len();
    if map_len != available {
        return Err(
            crate::storage::eligibility::EligibilityError::Inconsistent(format!(
                "eligibility map length {map_len} does not match vector count {available}"
            ))
            .into(),
        );
    }
    // Test hook for IO-error fallback (VAL-197-015 any-doubt): a sentinel path
    // triggers a typed IO doubt so the hybrid layer must fallback, not surface.
    if map.distinct_paths.iter().any(|p| p == "__io_test__") {
        return Err(crate::storage::eligibility::EligibilityError::Io(
            "simulated IO error for test".to_string(),
        )
        .into());
    }
    let max_k = limit.min(available);
    if max_k == 0 {
        return Ok(Vec::new());
    }
    let values = snapshot.data.as_slice();
    let mut heap = BinaryHeap::with_capacity(max_k);
    for (index, vector) in values.chunks_exact(dim).enumerate() {
        let rowid = index as i64 + 1;
        if is_tombstoned(&snapshot.tombstones, rowid) {
            continue;
        }
        if !crate::storage::eligibility::is_row_eligible(map, query_elig, index) {
            continue;
        }
        let distance = f32::euclidean(query, vector)
            .context("SimSIMD returned no distance for equal vector dimensions")?;
        let candidate = DistanceCandidate { rowid, distance };
        if heap.len() < max_k {
            heap.push(candidate);
        } else if let Some(top) = heap.peek()
            && candidate < *top
        {
            heap.pop();
            heap.push(candidate);
        }
    }

    let mut results: Vec<_> = heap.into_vec();
    results.sort();
    Ok(results)
}

fn bitmap_len(max_rowid: i64) -> usize {
    if max_rowid <= 0 {
        0
    } else {
        (max_rowid as usize).div_ceil(8)
    }
}

fn set_tombstone(bitmap: &mut [u8], rowid: i64, deleted: bool) -> Result<()> {
    if rowid <= 0 {
        anyhow::bail!("vector rowid must be positive")
    }
    let index = rowid as usize - 1;
    let byte = index / 8;
    let bit = 1u8 << (index % 8);
    let slot = bitmap
        .get_mut(byte)
        .context("vector rowid exceeds tombstone bitmap")?;
    if deleted {
        *slot |= bit;
    } else {
        *slot &= !bit;
    }
    Ok(())
}

fn is_tombstoned(bitmap: &[u8], rowid: i64) -> bool {
    if rowid <= 0 {
        return true;
    }
    let index = rowid as usize - 1;
    bitmap
        .get(index / 8)
        .is_some_and(|byte| byte & (1u8 << (index % 8)) != 0)
}

fn count_tombstones(bitmap: &[u8], max_rowid: i64) -> u64 {
    if max_rowid <= 0 {
        return 0;
    }
    let max_rowid = max_rowid as usize;
    let full_bytes = max_rowid / 8;
    let mut count: u64 = bitmap
        .get(..full_bytes)
        .unwrap_or_default()
        .iter()
        .map(|byte| u64::from(byte.count_ones()))
        .sum();
    if let Some(remainder) = max_rowid.checked_rem(8)
        && remainder != 0
        && let Some(byte) = bitmap.get(full_bytes)
    {
        count += u64::from((byte & ((1u8 << remainder) - 1)).count_ones());
    }
    count
}

fn resize_tombstones(bitmap: &mut Vec<u8>, old_max_rowid: i64, new_max_rowid: i64) -> Result<()> {
    if old_max_rowid < 0 || new_max_rowid < 0 || new_max_rowid < old_max_rowid {
        anyhow::bail!("invalid vector rowid range for tombstone bitmap")
    }
    bitmap.resize(bitmap_len(new_max_rowid), 0);
    for rowid in old_max_rowid + 1..=new_max_rowid {
        set_tombstone(bitmap, rowid, true)?;
    }
    clear_tombstone_tail(bitmap, new_max_rowid);
    Ok(())
}

fn clear_tombstone_tail(bitmap: &mut [u8], max_rowid: i64) {
    if max_rowid <= 0 {
        return;
    }
    let remainder = (max_rowid as usize) % 8;
    if remainder != 0
        && let Some(last) = bitmap.last_mut()
    {
        *last &= (1u8 << remainder) - 1;
    }
}

fn tombstones_are_consistent(bitmap: &[u8], max_rowid: i64) -> bool {
    if bitmap.len() != bitmap_len(max_rowid) {
        return false;
    }
    if max_rowid <= 0 {
        return bitmap.is_empty();
    }
    let remainder = (max_rowid as usize) % 8;
    remainder == 0
        || bitmap
            .last()
            .is_some_and(|last| *last & !((1u8 << remainder) - 1) == 0)
}

fn manifest_mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

fn current_generation(conn: &Connection) -> Result<u64> {
    let value: String = conn
        .query_row(
            "SELECT value FROM vector_store_meta WHERE key = 'generation'",
            [],
            |row| row.get(0),
        )
        .context("failed to read vector store generation")?;
    value
        .parse::<u64>()
        .with_context(|| format!("invalid vector store generation: {value}"))
}

fn current_database_id(conn: &Connection) -> Result<String> {
    conn.query_row(
        "SELECT value FROM vector_store_meta WHERE key = ?1",
        params![DATABASE_ID_KEY],
        |row| row.get(0),
    )
    .context("failed to read vector store database identity")
}

fn ensure_database_id(conn: &Connection) -> Result<()> {
    let existing: Option<String> = conn
        .query_row(
            "SELECT value FROM vector_store_meta WHERE key = ?1",
            params![DATABASE_ID_KEY],
            |row| row.get(0),
        )
        .optional()
        .context("failed to inspect vector store database identity")?;
    if existing.is_some_and(|value| !value.is_empty()) {
        return Ok(());
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = TEMP_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let database_id = format!("{timestamp:032x}-{sequence:016x}");
    conn.execute(
        "INSERT OR IGNORE INTO vector_store_meta (key, value) VALUES (?1, ?2)",
        params![DATABASE_ID_KEY, database_id],
    )
    .context("failed to initialize vector store database identity")?;
    Ok(())
}

fn bump_generation(tx: &rusqlite::Transaction<'_>) -> Result<u64> {
    let current: String = tx
        .query_row(
            "SELECT value FROM vector_store_meta WHERE key = 'generation'",
            [],
            |row| row.get(0),
        )
        .context("failed to read vector store generation")?;
    let next = current
        .parse::<u64>()
        .with_context(|| format!("invalid vector store generation: {current}"))?
        .checked_add(1)
        .context("vector store generation overflow")?;
    tx.execute(
        "UPDATE vector_store_meta SET value = ?1 WHERE key = 'generation'",
        params![next.to_string()],
    )
    .context("failed to update vector store generation")?;
    Ok(next)
}

fn current_max_rowid(conn: &Connection) -> Result<i64> {
    conn.query_row(
        "SELECT COALESCE(
            (SELECT seq FROM sqlite_sequence WHERE name = 'chunk_id_map'),
            (SELECT MAX(rowid) FROM chunk_id_map),
            0
        )",
        [],
        |row| row.get(0),
    )
    .context("failed to read maximum vector rowid")
}

fn expected_flat_bytes(dim: usize, max_rowid: i64) -> Result<u64> {
    let max_rowid = u64::try_from(max_rowid).context("negative maximum vector rowid")?;
    let dim = u64::try_from(dim).context("vector dimension does not fit in u64")?;
    max_rowid
        .checked_mul(dim)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .context("flat vector storage size overflow")
}

fn load_or_rebuild_disk_snapshot(
    conn: &Connection,
    dim: usize,
    paths: FlatPaths<'_>,
) -> Result<FlatSnapshot> {
    let generation = current_generation(conn)?;
    let database_id = current_database_id(conn)?;
    let max_rowid = current_max_rowid(conn)?;
    if let Some(snapshot) = load_valid_disk_snapshot(
        dim,
        &database_id,
        generation,
        max_rowid,
        paths.flat,
        paths.tombstone,
        paths.manifest,
    )? {
        return Ok(snapshot);
    }

    tracing::debug!(
        flat = %paths.flat.display(),
        "flat vector sidecar missing or inconsistent; rebuilding from vec0"
    );
    rebuild_disk_snapshot(conn, dim, &database_id, max_rowid, generation, paths)
}

fn load_valid_disk_snapshot(
    dim: usize,
    database_id: &str,
    generation: u64,
    max_rowid: i64,
    flat_path: &Path,
    tombstone_path: &Path,
    manifest_path: &Path,
) -> Result<Option<FlatSnapshot>> {
    let manifest_bytes = match fs::read(manifest_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            tracing::debug!(error = %error, "failed to read flat vector manifest");
            return Ok(None);
        }
    };
    let manifest: FlatManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(manifest) => manifest,
        Err(error) => {
            tracing::debug!(error = %error, "failed to parse flat vector manifest");
            return Ok(None);
        }
    };
    let expected_bytes = expected_flat_bytes(dim, max_rowid)?;
    let flat_size = match fs::metadata(flat_path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            tracing::debug!(error = %error, "failed to stat flat vector file");
            return Ok(None);
        }
    };
    let tombstone_bytes = match fs::read(tombstone_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            tracing::debug!(error = %error, "failed to read flat vector tombstones");
            return Ok(None);
        }
    };
    let expected_tombstones = bitmap_len(max_rowid);
    if manifest.version != FLAT_MANIFEST_VERSION
        || manifest.database_id != database_id
        || manifest.dim != dim
        || manifest.max_rowid != max_rowid
        || manifest.generation != generation
        || manifest.flat_bytes != expected_bytes
        || flat_size != expected_bytes
        || tombstone_bytes.len() != expected_tombstones
        || !tombstones_are_consistent(&tombstone_bytes, max_rowid)
        || manifest.tombstone_count != count_tombstones(&tombstone_bytes, max_rowid)
    {
        return Ok(None);
    }

    let data = if expected_bytes == 0 {
        FlatData::Empty
    } else {
        let file = File::open(flat_path).context("failed to open flat vector file")?;
        // SAFETY: the file is validated against the manifest before mapping and
        // is replaced atomically by writers, so this mapping's inode remains
        // stable for its lifetime.
        FlatData::Mmap(unsafe { Mmap::map(&file) }.context("failed to mmap flat vector file")?)
    };
    Ok(Some(FlatSnapshot {
        manifest,
        data,
        tombstones: tombstone_bytes,
        manifest_mtime: manifest_mtime(manifest_path),
    }))
}

fn rebuild_disk_snapshot(
    conn: &Connection,
    dim: usize,
    database_id: &str,
    max_rowid: i64,
    generation: u64,
    paths: FlatPaths<'_>,
) -> Result<FlatSnapshot> {
    let max_rowid_usize =
        usize::try_from(max_rowid).context("maximum vector rowid out of range")?;
    let values_len = max_rowid_usize
        .checked_mul(dim)
        .context("flat vector storage size overflow")?;
    let mut values = vec![0.0f32; values_len];
    let mut tombstones = vec![u8::MAX; bitmap_len(max_rowid)];
    clear_tombstone_tail(&mut tombstones, max_rowid);
    let mut stmt = conn
        .prepare("SELECT rowid, embedding FROM vec_chunks ORDER BY rowid")
        .context("failed to prepare flat vector rebuild")?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .context("failed to query vectors for flat rebuild")?;
    for row in rows {
        let (rowid, bytes) = row.context("failed to read vector for flat rebuild")?;
        let rowid_usize = usize::try_from(rowid).context("vector rowid out of range")?;
        if rowid_usize == 0 || rowid_usize > max_rowid_usize {
            anyhow::bail!("vector rowid {rowid} exceeds the SQLite rowid space")
        }
        let vector = decode_f32_vector(&bytes, dim)
            .with_context(|| format!("failed to decode vector rowid {rowid}"))?;
        let start = (rowid_usize - 1)
            .checked_mul(dim)
            .context("flat vector storage offset overflow")?;
        values[start..start + dim].copy_from_slice(&vector);
        set_tombstone(&mut tombstones, rowid, false)?;
    }
    let manifest = FlatManifest {
        version: FLAT_MANIFEST_VERSION,
        database_id: database_id.to_string(),
        dim,
        max_rowid,
        generation,
        tombstone_count: count_tombstones(&tombstones, max_rowid),
        flat_bytes: expected_flat_bytes(dim, max_rowid)?,
    };
    publish_disk_files(
        paths.flat,
        paths.tombstone,
        paths.manifest,
        &values,
        &tombstones,
        &manifest,
    )?;
    load_valid_disk_snapshot(
        dim,
        database_id,
        generation,
        max_rowid,
        paths.flat,
        paths.tombstone,
        paths.manifest,
    )?
    .context("flat vector rebuild did not produce a valid snapshot")
}

fn decode_f32_vector(bytes: &[u8], dim: usize) -> Result<Vec<f32>> {
    let expected = dim
        .checked_mul(std::mem::size_of::<f32>())
        .context("vector byte size overflow")?;
    if bytes.len() != expected {
        anyhow::bail!(
            "vector byte length mismatch: expected {}, got {}",
            expected,
            bytes.len()
        );
    }
    Ok(bytes
        .as_chunks::<{ std::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn apply_disk_update(
    conn: &Connection,
    storage: &DiskFlatStorage,
    inserts: &[(i64, &[f32])],
    tombstone_rowids: &[i64],
    generation: u64,
) -> Result<()> {
    let max_rowid = current_max_rowid(conn)?;
    let max_rowid_usize =
        usize::try_from(max_rowid).context("maximum vector rowid out of range")?;
    let previous_max_rowid = storage.snapshot.manifest.max_rowid;
    if max_rowid < previous_max_rowid {
        anyhow::bail!(
            "flat vector maximum rowid moved backwards: previous {}, current {}",
            previous_max_rowid,
            max_rowid
        );
    }
    let expected_bytes = expected_flat_bytes(storage.dim, max_rowid)?;
    let previous_bytes = expected_flat_bytes(storage.dim, previous_max_rowid)?;
    let actual_bytes = fs::metadata(&storage.flat_path)
        .with_context(|| {
            format!(
                "failed to stat flat vector file: {}",
                storage.flat_path.display()
            )
        })?
        .len();
    if actual_bytes != previous_bytes {
        anyhow::bail!(
            "flat vector file size mismatch before incremental update: expected {}, got {}",
            previous_bytes,
            actual_bytes
        );
    }

    let mut writes = Vec::with_capacity(inserts.len());
    for (rowid, vector) in inserts {
        if vector.len() != storage.dim {
            anyhow::bail!(
                "vector dimension mismatch: expected {}, got {}",
                storage.dim,
                vector.len()
            );
        }
        let rowid_usize = usize::try_from(*rowid).context("vector rowid out of range")?;
        if rowid_usize == 0 || rowid_usize > max_rowid_usize {
            anyhow::bail!("vector rowid {rowid} exceeds the SQLite rowid space")
        }
        writes.push((vector_byte_offset(*rowid, storage.dim)?, *vector));
    }

    let mut flat_file = OpenOptions::new()
        .write(true)
        .open(&storage.flat_path)
        .with_context(|| {
            format!(
                "failed to open flat vector file for update: {}",
                storage.flat_path.display()
            )
        })?;
    for (offset, vector) in writes {
        write_vector_at(&mut flat_file, offset, vector)?;
    }
    flat_file.set_len(expected_bytes).with_context(|| {
        format!(
            "failed to size flat vector file: {}",
            storage.flat_path.display()
        )
    })?;
    flat_file.sync_all().with_context(|| {
        format!(
            "failed to sync flat vector file: {}",
            storage.flat_path.display()
        )
    })?;

    let mut tombstones = storage.snapshot.tombstones.clone();
    resize_tombstones(
        &mut tombstones,
        storage.snapshot.manifest.max_rowid,
        max_rowid,
    )?;
    for rowid in tombstone_rowids {
        set_tombstone(&mut tombstones, *rowid, true)?;
    }
    for (rowid, _) in inserts {
        set_tombstone(&mut tombstones, *rowid, false)?;
    }
    let manifest = FlatManifest {
        version: FLAT_MANIFEST_VERSION,
        database_id: current_database_id(conn)?,
        dim: storage.dim,
        max_rowid,
        generation,
        tombstone_count: count_tombstones(&tombstones, max_rowid),
        flat_bytes: expected_bytes,
    };
    let tombstone_tmp = write_temp_file(&storage.tombstone_path, &tombstones)?;
    let manifest_bytes = serde_json::to_vec(&manifest).context("failed to encode flat manifest")?;
    let manifest_tmp = match write_temp_file(&storage.manifest_path, &manifest_bytes) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_file(&tombstone_tmp);
            return Err(error);
        }
    };

    if let Err(error) = fs::rename(&tombstone_tmp, &storage.tombstone_path) {
        let _ = fs::remove_file(&tombstone_tmp);
        let _ = fs::remove_file(&manifest_tmp);
        return Err(error).context("failed to publish vector tombstones");
    }
    if let Err(error) = fs::rename(&manifest_tmp, &storage.manifest_path) {
        let _ = fs::remove_file(&manifest_tmp);
        return Err(error).with_context(|| {
            format!(
                "failed to publish vector manifest: {}",
                storage.manifest_path.display()
            )
        });
    }
    Ok(())
}

fn vector_byte_offset(rowid: i64, dim: usize) -> Result<u64> {
    let rowid = u64::try_from(rowid)
        .context("vector rowid out of range")?
        .checked_sub(1)
        .context("vector rowid must be positive")?;
    rowid
        .checked_mul(u64::try_from(dim).context("vector dimension does not fit in u64")?)
        .and_then(|offset| offset.checked_mul(std::mem::size_of::<f32>() as u64))
        .context("flat vector storage offset overflow")
}

fn write_vector_at(file: &mut File, offset: u64, vector: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("failed to seek flat vector file to byte {offset}"))?;
    file.write_all(&bytes)
        .with_context(|| format!("failed to write flat vector row at byte {offset}"))
}

fn publish_disk_files(
    flat_path: &Path,
    tombstone_path: &Path,
    manifest_path: &Path,
    values: &[f32],
    tombstones: &[u8],
    manifest: &FlatManifest,
) -> Result<()> {
    let mut flat_bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        flat_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let manifest_bytes = serde_json::to_vec(manifest).context("failed to encode flat manifest")?;
    let flat_tmp = write_temp_file(flat_path, &flat_bytes)?;
    let tombstone_tmp = match write_temp_file(tombstone_path, tombstones) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_file(&flat_tmp);
            return Err(error);
        }
    };
    let manifest_tmp = match write_temp_file(manifest_path, &manifest_bytes) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_file(&flat_tmp);
            let _ = fs::remove_file(&tombstone_tmp);
            return Err(error);
        }
    };

    if let Err(error) = fs::rename(&flat_tmp, flat_path) {
        let _ = fs::remove_file(&flat_tmp);
        let _ = fs::remove_file(&tombstone_tmp);
        let _ = fs::remove_file(&manifest_tmp);
        return Err(error).context("failed to publish flat vector file");
    }
    if let Err(error) = fs::rename(&tombstone_tmp, tombstone_path) {
        let _ = fs::remove_file(&tombstone_tmp);
        let _ = fs::remove_file(&manifest_tmp);
        return Err(error).context("failed to publish vector tombstones");
    }
    fs::rename(&manifest_tmp, manifest_path).with_context(|| {
        format!(
            "failed to publish vector manifest: {}",
            manifest_path.display()
        )
    })
}

fn write_temp_file(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp_path = path.with_extension(format!(
        "{}-{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("sidecar"),
        counter
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .with_context(|| {
            format!(
                "failed to create sidecar temp file: {}",
                temp_path.display()
            )
        })?;
    file.write_all(bytes)
        .with_context(|| format!("failed to write sidecar temp file: {}", temp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to sync sidecar temp file: {}", temp_path.display()))?;
    Ok(temp_path)
}

/// Dual-backed vector store for embedding search.
///
/// SQLite vec0 remains the durable source used for writes and sidecar
/// recovery. The default search path scans the exact flat SIMD sidecar; set
/// `VERA_VECTOR_SCAN=vec0` to select sqlite-vec at open time.
pub struct VectorStore {
    conn: Connection,
    dim: usize,
    scan_mode: VectorScanMode,
    flat: Mutex<FlatStorage>,
}

/// Maximum `k` sqlite-vec (vec0) accepts in a KNN query. Requesting more is a hard
/// error from the extension, not a soft limit. The flat scanner is not bounded
/// by this cap and honors the caller's requested depth up to the actual vector
/// count (see `scan_snapshot`).
///
/// Public so callers can size their candidate pools against the real vec0 ceiling
/// instead of scaling past it and relying on [`VectorStore::search`] to clamp.
pub const MAX_KNN_K: usize = 4096;

/// A single vector search result: chunk ID and distance score.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// The chunk ID (rowid in the vec table, mapped to chunk string ID).
    pub chunk_id: String,
    /// Distance from the query vector (lower is closer).
    pub distance: f64,
}

impl VectorStore {
    /// Open (or create) a vector store at the given path.
    ///
    /// The `dim` parameter specifies the vector dimensionality.
    pub fn open(db_path: &Path, dim: usize) -> Result<Self> {
        Self::open_at_mode(db_path, dim, VectorScanMode::from_env())
    }

    fn open_at_mode(db_path: &Path, dim: usize, scan_mode: VectorScanMode) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open(db_path)
            .with_context(|| format!("failed to open vector db: {}", db_path.display()))?;
        let sidecar_dir = db_path.parent().filter(|path| !path.as_os_str().is_empty());
        let sidecar_dir = sidecar_dir.unwrap_or_else(|| Path::new("."));
        Self::from_connection(conn, dim, scan_mode, Some(sidecar_dir))
    }

    /// Create an in-memory vector store (useful for testing).
    pub fn open_in_memory(dim: usize) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().context("failed to open in-memory vector db")?;
        Self::from_connection(conn, dim, VectorScanMode::from_env(), None)
    }

    fn from_connection(
        conn: Connection,
        dim: usize,
        scan_mode: VectorScanMode,
        sidecar_dir: Option<&Path>,
    ) -> Result<Self> {
        let store = Self {
            conn,
            dim,
            scan_mode,
            flat: Mutex::new(FlatStorage::Memory(MemoryFlatStorage {
                dim,
                snapshot: FlatSnapshot::empty(dim, 0),
            })),
        };
        store.init_schema()?;
        if let Some(sidecar_dir) = sidecar_dir {
            let flat = FlatStorage::open_disk(&store.conn, dim, sidecar_dir)?;
            *store
                .flat
                .lock()
                .map_err(|_| anyhow::anyhow!("flat vector storage lock poisoned"))? = flat;
        }
        Ok(store)
    }

    #[cfg(test)]
    fn open_in_memory_with_mode(dim: usize, scan_mode: VectorScanMode) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().context("failed to open in-memory vector db")?;
        Self::from_connection(conn, dim, scan_mode, None)
    }

    #[cfg(test)]
    fn open_disk_with_mode(db_path: &Path, dim: usize, scan_mode: VectorScanMode) -> Result<Self> {
        Self::open_at_mode(db_path, dim, scan_mode)
    }

    #[cfg(test)]
    fn rowids_for_chunk_ids(&self, ids: &[&str]) -> Vec<i64> {
        ids.iter()
            .map(|id| {
                self.conn
                    .query_row(
                        "SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1",
                        params![id],
                        |row| row.get(0),
                    )
                    .unwrap()
            })
            .collect()
    }

    /// Initialize the vector table schema.
    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .context("failed to set vector db pragmas")?;

        // Mapping from string chunk IDs to integer rowids.
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS chunk_id_map (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE
                );
                -- `chunk_id` is UNIQUE, which already builds
                -- `sqlite_autoindex_chunk_id_map_1` over the same column. The
                -- explicit index duplicated it exactly and only added a second
                -- B-tree to maintain on every vector insert. Dropped here so
                -- existing databases shed it on next open; the autoindex
                -- serves both the equality lookup and the prefix range scan.
                DROP INDEX IF EXISTS idx_chunk_id_map;",
            )
            .context("failed to create chunk_id_map table")?;

        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS vector_store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO vector_store_meta (key, value)
                    VALUES ('generation', '0');",
            )
            .context("failed to create vector store metadata")?;
        ensure_database_id(&self.conn)?;

        // sqlite-vec virtual table for vector storage.
        self.conn
            .execute_batch(&format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                 USING vec0(embedding float[{}])",
                self.dim
            ))
            .context("failed to create vec_chunks virtual table")?;

        Ok(())
    }

    /// Insert a single vector for a chunk.
    ///
    /// Uses INSERT OR IGNORE + SELECT to get a stable rowid, avoiding the
    /// AUTOINCREMENT orphan problem where INSERT OR REPLACE allocates a
    /// new rowid and orphans old vectors in the vec_chunks virtual table.
    /// For re-inserts (same chunk_id), deletes the old vector first since
    /// the vec0 virtual table does not support INSERT OR REPLACE.
    pub fn insert(&self, chunk_id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            anyhow::bail!(
                "vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            );
        }

        self.refresh_flat()?;

        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )
        .context("failed to begin vector insert transaction")?;
        // Use INSERT OR IGNORE to preserve existing rowid if already present.
        tx.execute(
            "INSERT OR IGNORE INTO chunk_id_map (chunk_id) VALUES (?1)",
            params![chunk_id],
        )
        .context("failed to insert chunk id mapping")?;

        let rowid: i64 = tx
            .query_row(
                "SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1",
                params![chunk_id],
                |row| row.get(0),
            )
            .context("failed to get rowid for chunk")?;

        // Delete any existing vector for this rowid before inserting.
        // vec0 virtual tables do not support INSERT OR REPLACE. Deleting an
        // absent rowid is Ok with 0 rows, so only genuine failures (I/O,
        // virtual-table malfunction) reach this error path; swallowing them
        // would persist a mapping row the count and KNN cannot agree on.
        tx.execute("DELETE FROM vec_chunks WHERE rowid = ?1", params![rowid])
            .context("failed to delete stale vector")?;

        tx.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?1, ?2)",
            params![rowid, vector.as_bytes()],
        )
        .context("failed to insert vector")?;

        let generation = bump_generation(&tx)?;
        tx.commit().context("failed to commit vector insert")?;
        self.update_flat(&[(rowid, vector)], &[], generation)?;
        Ok(())
    }

    /// Insert a batch of vectors.
    ///
    /// Uses INSERT OR IGNORE to preserve stable rowids, avoiding the
    /// AUTOINCREMENT orphan problem. For re-inserts, deletes old vectors
    /// first since the vec0 virtual table doesn't support upsert.
    pub fn insert_batch(&self, items: &[(&str, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.refresh_flat()?;
        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin vector insert transaction")?;
        let mut flat_inserts = Vec::with_capacity(items.len());
        {
            let mut id_stmt = self
                .conn
                .prepare_cached("INSERT OR IGNORE INTO chunk_id_map (chunk_id) VALUES (?1)")
                .context("failed to prepare id insert")?;

            let mut rowid_stmt = self
                .conn
                .prepare_cached("SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1")
                .context("failed to prepare rowid query")?;

            let mut del_vec_stmt = self
                .conn
                .prepare_cached("DELETE FROM vec_chunks WHERE rowid = ?1")
                .context("failed to prepare vector delete")?;

            let mut vec_stmt = self
                .conn
                .prepare_cached("INSERT INTO vec_chunks (rowid, embedding) VALUES (?1, ?2)")
                .context("failed to prepare vector insert")?;

            for (chunk_id, vector) in items {
                if vector.len() != self.dim {
                    anyhow::bail!(
                        "vector dimension mismatch for {}: expected {}, got {}",
                        chunk_id,
                        self.dim,
                        vector.len()
                    );
                }

                id_stmt
                    .execute(params![chunk_id])
                    .context("failed to insert chunk id")?;

                let rowid: i64 = rowid_stmt
                    .query_row(params![chunk_id], |row| row.get(0))
                    .context("failed to get rowid")?;

                // Same as in insert(): deleting an absent rowid is Ok with 0
                // rows, so only genuine failures reach the error path.
                del_vec_stmt
                    .execute(params![rowid])
                    .context("failed to delete stale vector")?;

                vec_stmt
                    .execute(params![rowid, vector.as_bytes()])
                    .context("failed to insert vector")?;
                flat_inserts.push((rowid, *vector));
            }
        }
        let generation = bump_generation(&tx)?;
        tx.commit().context("failed to commit vector batch")?;
        self.update_flat(&flat_inserts, &[], generation)?;
        Ok(())
    }

    /// Insert a batch of vectors whose chunk ids are all new to this store.
    ///
    /// Full builds write into a freshly created staging store, where the
    /// per-row `INSERT OR IGNORE` + rowid lookup + stale-vector delete that
    /// [`VectorStore::insert_batch`] performs can never match anything. This
    /// path halves the per-row statement count for that case. A pre-existing
    /// chunk id violates the PRIMARY KEY and fails loudly, so callers cannot
    /// silently corrupt an existing mapping.
    pub fn insert_batch_fresh(&self, items: &[(&str, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.refresh_flat()?;
        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin vector insert transaction")?;
        let mut flat_inserts = Vec::with_capacity(items.len());
        {
            let mut id_stmt = self
                .conn
                .prepare_cached("INSERT INTO chunk_id_map (chunk_id) VALUES (?1)")
                .context("failed to prepare id insert")?;

            let mut vec_stmt = self
                .conn
                .prepare_cached("INSERT INTO vec_chunks (rowid, embedding) VALUES (?1, ?2)")
                .context("failed to prepare vector insert")?;

            for (chunk_id, vector) in items {
                if vector.len() != self.dim {
                    anyhow::bail!(
                        "vector dimension mismatch for {}: expected {}, got {}",
                        chunk_id,
                        self.dim,
                        vector.len()
                    );
                }

                id_stmt
                    .execute(params![chunk_id])
                    .context("failed to insert chunk id")?;
                let rowid = self.conn.last_insert_rowid();

                vec_stmt
                    .execute(params![rowid, vector.as_bytes()])
                    .context("failed to insert vector")?;
                flat_inserts.push((rowid, *vector));
            }
        }
        let generation = bump_generation(&tx)?;
        tx.commit().context("failed to commit vector batch")?;
        self.update_flat(&flat_inserts, &[], generation)?;
        Ok(())
    }

    /// Find the nearest neighbors to a query vector.
    ///
    /// Returns up to `limit` results sorted by ascending distance.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dim {
            anyhow::bail!(
                "query vector dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            );
        }

        if self.scan_mode == VectorScanMode::Flat {
            let mut flat = self
                .flat
                .lock()
                .map_err(|_| anyhow::anyhow!("flat vector storage lock poisoned"))?;
            let hits = flat.search(&self.conn, query, limit)?;
            let rowids: Vec<i64> = hits.iter().map(|hit| hit.rowid).collect();
            let chunk_ids = self.chunk_ids_for_rowids(&rowids)?;
            return hits
                .iter()
                .map(|hit| {
                    let chunk_id = chunk_ids
                        .get(&hit.rowid)
                        .with_context(|| format!("failed to map rowid {} to chunk_id", hit.rowid))?
                        .clone();
                    Ok(VectorSearchResult {
                        chunk_id,
                        distance: hit.distance,
                    })
                })
                .collect();
        }

        // sqlite-vec reads this LIMIT as the KNN `k` and rejects anything above
        // MAX_KNN_K with "k value in knn query too large". Callers scale the
        // candidate pool from the query type and result limit, which can exceed
        // it on natural-language queries, and the whole vector arm would then be
        // dropped in favour of BM25-only results. Ask for as many as the backend
        // allows instead.
        //
        // Truncation diagnostics are intentionally not emitted here: this storage
        // layer has no filter context, so a generic `limit > cap` warn would fire
        // on filtered true negatives (e.g. `--path src/does-not-exist` on a
        // 4427-chunk index) while the hybrid layer correctly stays quiet via
        // `has_filter_matches` (which now includes `scope`/`include_generated`).
        // Filter-aware, actionable diagnostics live in `retrieval::hybrid`.
        let limit = limit.min(MAX_KNN_K);

        // `prepare`, not `prepare_cached`: `limit` is interpolated into the
        // text, so every distinct limit is a distinct cache key. Caching these
        // never hits and evicts the statements that do have stable text, since
        // rusqlite's cache is a 16-entry LRU shared by the whole connection.
        let mut stmt = self
            .conn
            .prepare(&format!(
                "SELECT v.rowid, v.distance
                 FROM vec_chunks v
                 WHERE v.embedding MATCH ?1
                 ORDER BY v.distance
                 LIMIT {limit}"
            ))
            .context("failed to prepare vector search")?;

        let hits: Vec<(i64, f64)> = stmt
            .query_map([query.as_bytes()], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
            })
            .context("failed to execute vector search")?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read vector result")?;

        // Resolve the rowids in one query instead of one per hit. The KNN
        // query above is deliberately left alone: joining `chunk_id_map` into
        // it would risk losing the vec0 KNN optimization.
        let rowids: Vec<i64> = hits.iter().map(|(rowid, _)| *rowid).collect();
        let chunk_ids = self.chunk_ids_for_rowids(&rowids)?;

        hits.iter()
            .map(|(rowid, distance)| {
                let chunk_id = chunk_ids
                    .get(rowid)
                    .with_context(|| format!("failed to map rowid {rowid} to chunk_id"))?
                    .clone();
                Ok(VectorSearchResult {
                    chunk_id,
                    distance: *distance,
                })
            })
            .collect()
    }

    /// Filtered flat search using the eligibility map (filter-during-scan).
    /// Returns up to `limit` eligible results sorted by distance. Callers must
    /// have already verified `is_flat` and that the filter set is map-evaluable.
    /// An empty eligibility (no match) returns empty without scanning.
    pub fn search_filtered(
        &self,
        query: &[f32],
        limit: usize,
        map: &crate::storage::eligibility::EligibilityMap,
        query_elig: &crate::storage::eligibility::QueryEligibility,
    ) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dim {
            anyhow::bail!(
                "query vector dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            );
        }
        if query_elig.is_empty() {
            return Ok(Vec::new());
        }
        if self.scan_mode != VectorScanMode::Flat {
            anyhow::bail!("filtered scan is only available on the flat backend");
        }
        let mut flat = self
            .flat
            .lock()
            .map_err(|_| anyhow::anyhow!("flat vector storage lock poisoned"))?;
        let hits = flat.search_filtered(&self.conn, query, limit, map, query_elig)?;
        let rowids: Vec<i64> = hits.iter().map(|hit| hit.rowid).collect();
        let chunk_ids = self.chunk_ids_for_rowids(&rowids)?;
        hits.iter()
            .map(|hit| {
                let chunk_id = chunk_ids
                    .get(&hit.rowid)
                    .with_context(|| format!("failed to map rowid {} to chunk_id", hit.rowid))?
                    .clone();
                Ok(VectorSearchResult {
                    chunk_id,
                    distance: hit.distance,
                })
            })
            .collect()
    }

    /// Resolve many rowids to their chunk ids in a single query, batching to stay
    /// below SQLite's variable limit. Result order is caller's responsibility.
    fn chunk_ids_for_rowids(&self, rowids: &[i64]) -> Result<HashMap<i64, String>> {
        if rowids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut mapped = HashMap::with_capacity(rowids.len());
        for batch in rowids.chunks(crate::storage::SQL_PARAMETER_BATCH) {
            let placeholders = crate::storage::sql_placeholders(batch.len());
            // Text varies with the number of ids, so plain `prepare` here too.
            let mut stmt = self
                .conn
                .prepare(&format!(
                    "SELECT rowid, chunk_id FROM chunk_id_map WHERE rowid IN ({placeholders})"
                ))
                .context("failed to prepare chunk id lookup")?;

            let batch_map = stmt
                .query_map(rusqlite::params_from_iter(batch.iter()), |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .context("failed to query chunk ids")?
                .collect::<std::result::Result<HashMap<_, _>, _>>()
                .context("failed to collect chunk ids")?;
            mapped.extend(batch_map);
        }
        Ok(mapped)
    }

    /// Count vectors actually stored, and therefore searchable.
    ///
    /// `chunk_id_map` historically backed this number, but a mapping row can
    /// exist without a backing vector in databases written before single
    /// inserts became transactional. KNN resolves every hit through
    /// `vec_chunks`, so that is what an honest count reports; orphan mappings
    /// neither inflate it nor surface in searches, and they disappear when
    /// their file is next re-indexed.
    pub fn count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
            .context("failed to count vectors")?;
        Ok(count as u64)
    }

    /// Whether this store is using the flat (SIMD) scan path.
    pub fn is_flat(&self) -> bool {
        self.scan_mode == VectorScanMode::Flat
    }

    /// Whether this store is using the vec0 (sqlite-vec) path.
    pub fn is_vec0(&self) -> bool {
        self.scan_mode == VectorScanMode::Vec0
    }

    /// Delete a vector by chunk ID.
    pub fn delete(&self, chunk_id: &str) -> Result<bool> {
        let rowid: Option<i64> = self
            .conn
            .query_row(
                "SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1",
                params![chunk_id],
                |row| row.get(0),
            )
            .optional()
            .context("failed to look up chunk for deletion")?;

        if let Some(rowid) = rowid {
            self.refresh_flat()?;
            let tx = rusqlite::Transaction::new_unchecked(
                &self.conn,
                rusqlite::TransactionBehavior::Immediate,
            )
            .context("failed to begin vector delete transaction")?;
            tx.execute("DELETE FROM vec_chunks WHERE rowid = ?1", params![rowid])
                .context("failed to delete vector")?;
            tx.execute("DELETE FROM chunk_id_map WHERE rowid = ?1", params![rowid])
                .context("failed to delete chunk id mapping")?;
            let generation = bump_generation(&tx)?;
            tx.commit().context("failed to commit vector delete")?;
            self.update_flat(&[], &[rowid], generation)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete all vectors whose chunk_id starts with the given prefix.
    ///
    /// This is used for incremental indexing: when a file is re-indexed, all
    /// old chunks for that file (whose IDs start with "filepath:") are removed.
    pub fn delete_by_file_prefix(&self, prefix: &str) -> Result<u64> {
        self.delete_by_file_prefix_after_scan(prefix, || {})
    }

    fn delete_by_file_prefix_after_scan<F>(&self, prefix: &str, after_scan: F) -> Result<u64>
    where
        F: FnOnce(),
    {
        self.refresh_flat()?;

        // A half-open range rather than `LIKE ?1 ESCAPE '\'`. The ESCAPE
        // clause disqualifies SQLite's LIKE-prefix optimization, so the LIKE
        // form scans `chunk_id_map` in full where a range seeks the index. It
        // also removes the need to escape `%` and `_`, since a range has no
        // wildcards to confuse.
        // One transaction for the whole file instead of two commits per row,
        // opened *before* the scan so a concurrent writer on the same database
        // cannot insert a matching row between finding the rowids and deleting
        // them — that row would survive while this reported success.
        //
        // IMMEDIATE rather than the default DEFERRED because this transaction
        // reads and then writes: a deferred transaction starts as a reader and
        // can fail the upgrade with SQLITE_BUSY_SNAPSHOT, which `busy_timeout`
        // does not retry.
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )
        .context("failed to begin prefix delete transaction")?;

        let rows = rowids_with_prefix(&tx, prefix)?;
        after_scan();
        let count = rows.len() as u64;
        if count == 0 {
            return Ok(count);
        }

        {
            let mut delete_vector = tx
                .prepare_cached("DELETE FROM vec_chunks WHERE rowid = ?1")
                .context("failed to prepare vector delete")?;
            let mut delete_mapping = tx
                .prepare_cached("DELETE FROM chunk_id_map WHERE rowid = ?1")
                .context("failed to prepare chunk id delete")?;
            for rowid in &rows {
                delete_vector
                    .execute(params![rowid])
                    .context("failed to delete vector by prefix")?;
                delete_mapping
                    .execute(params![rowid])
                    .context("failed to delete chunk id by prefix")?;
            }
        }
        let generation = bump_generation(&tx)?;
        tx.commit().context("failed to commit prefix delete")?;
        self.update_flat(&[], &rows, generation)?;

        Ok(count)
    }

    /// Clear all vectors from the store.
    pub fn clear(&self) -> Result<()> {
        self.refresh_flat()?;
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )
        .context("failed to begin vector clear transaction")?;
        tx.execute_batch("DELETE FROM vec_chunks; DELETE FROM chunk_id_map;")
            .context("failed to clear vector store")?;
        let generation = bump_generation(&tx)?;
        let max_rowid = current_max_rowid(&self.conn)?;
        let tombstones: Vec<i64> = (1..=max_rowid).collect();
        tx.commit().context("failed to commit vector clear")?;
        self.update_flat(&[], &tombstones, generation)?;
        Ok(())
    }

    /// Get the configured vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn update_flat(
        &self,
        inserts: &[(i64, &[f32])],
        tombstone_rowids: &[i64],
        generation: u64,
    ) -> Result<()> {
        let mut flat = self
            .flat
            .lock()
            .map_err(|_| anyhow::anyhow!("flat vector storage lock poisoned"))?;
        if let FlatStorage::Disk(storage) = &*flat {
            let current_generation = current_generation(&self.conn)?;
            let current_database_id = current_database_id(&self.conn)?;
            let generation_changed = current_generation != generation;
            let database_changed = storage.snapshot.manifest.database_id != current_database_id;
            let snapshot_is_stale =
                generation.checked_sub(1) != Some(storage.snapshot.manifest.generation);
            if generation_changed || database_changed || snapshot_is_stale {
                tracing::debug!(
                    expected_generation = generation,
                    "vector store changed while publishing flat sidecar; reloading"
                );
                flat.reload_disk(&self.conn)?;
                return Ok(());
            }
        }
        flat.apply_update(&self.conn, inserts, tombstone_rowids, generation)
    }

    fn refresh_flat(&self) -> Result<()> {
        let mut flat = self
            .flat
            .lock()
            .map_err(|_| anyhow::anyhow!("flat vector storage lock poisoned"))?;
        flat.refresh(&self.conn)
    }
}

/// Rowids of every `chunk_id` starting with `prefix`, via a range scan.
///
/// Takes the connection rather than `&self` so the caller can pass an open
/// transaction, keeping the scan atomic with the deletes that follow it.
fn rowids_with_prefix(conn: &Connection, prefix: &str) -> Result<Vec<i64>> {
    // The two predicates stay distinct because `prefix_upper_bound` returns
    // None when nothing can sort above the prefix, and `>= prefix` is then
    // already exact. Only the execution and collection are shared; merging the
    // SQL would cost the index range plan, which is the point of all this.
    let upper = prefix_upper_bound(prefix);
    let (sql, args): (&str, Vec<&dyn rusqlite::ToSql>) = match &upper {
        Some(upper) => (PREFIX_RANGE_SQL, vec![&prefix, upper]),
        None => (PREFIX_LOWER_BOUND_SQL, vec![&prefix]),
    };

    let mut stmt = conn
        .prepare_cached(sql)
        .context("failed to prepare prefix delete query")?;
    let rows = stmt
        .query_map(args.as_slice(), |row| row.get::<_, i64>(0))
        .context("failed to query chunks by prefix")?
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to collect prefix results")?;
    Ok(rows)
}

/// Smallest string that sorts strictly above every string starting with
/// `prefix`, giving the exclusive upper bound of a prefix range scan.
///
/// SQLite's default `BINARY` collation compares TEXT bytewise, and UTF-8 byte
/// order matches code point order, so incrementing the final character is
/// sufficient. Two cases need care: the successor of a code point may land in
/// the surrogate range, which is not a valid `char`, so it is skipped; and a
/// trailing `char::MAX` has no successor at all, so it is dropped and the
/// character before it is incremented instead.
///
/// Returns `None` when no upper bound exists (an empty prefix, or one made
/// entirely of `char::MAX`), in which case `chunk_id >= prefix` is already
/// exact and needs no upper bound.
fn prefix_upper_bound(prefix: &str) -> Option<String> {
    let mut chars: Vec<char> = prefix.chars().collect();
    while let Some(last) = chars.pop() {
        if let Some(next) = next_char(last) {
            let mut bound: String = chars.into_iter().collect();
            bound.push(next);
            return Some(bound);
        }
    }
    None
}

/// Next valid `char` above `c`, skipping the UTF-16 surrogate range.
fn next_char(c: char) -> Option<char> {
    let mut code = c as u32 + 1;
    while code <= char::MAX as u32 {
        if let Some(next) = char::from_u32(code) {
            return Some(next);
        }
        code += 1;
    }
    None
}

/// Register the sqlite-vec extension globally (idempotent).
fn register_sqlite_vec() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe {
            // sqlite-vec requires registering via auto_extension with a transmute
            // from the C-style init function pointer to the sqlite3 extension type.
            #[allow(clippy::missing_transmute_annotations)]
            let func = std::mem::transmute(sqlite3_vec_init as *const ());
            sqlite3_auto_extension(Some(func));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random for testing.
        let mut v = Vec::with_capacity(dim);
        let mut s = seed;
        for _ in 0..dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.push(((s >> 33) as f32) / (u32::MAX as f32));
        }
        v
    }

    #[test]
    fn insert_and_count() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("chunk1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert("chunk2", &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let result = store.insert("chunk1", &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("dimension mismatch")
        );
    }

    #[test]
    fn nearest_neighbor_self_query() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        store.insert("c1", &v1).unwrap();
        store.insert("c2", &v2).unwrap();
        store.insert("c3", &v3).unwrap();

        // Query with v1 should return c1 as the closest.
        let results = store.search(&v1, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001); // Self-match should be ~0 distance.
    }

    #[test]
    fn search_pairs_each_chunk_id_with_its_own_distance() {
        // The rowid -> chunk_id mapping is now one batched query returning a
        // HashMap, so the results have to be re-projected back into the KNN
        // distance order. Insert so that rowid order (insertion order) is the
        // reverse of distance order, then assert the pairing — not just that
        // distances ascend, which holds for any implementation that emits one
        // result per hit in hit order.
        // Distances must be strictly increasing, or the KNN order between two
        // equidistant vectors is unspecified and the assertion below would be
        // decided by tie-breaking rather than by distance.
        //   near 0.0   mid ~0.894   far ~1.414   (L2 from the query)
        // Inserted in reverse of that order, so rowid order is not distance
        // order and a SQL-ordered result cannot pass by accident.
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("far", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        store.insert("mid", &[0.6, 0.8, 0.0, 0.0]).unwrap();
        store.insert("near", &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);

        let order: Vec<&str> = results.iter().map(|r| r.chunk_id.as_str()).collect();
        assert_eq!(
            order,
            vec!["near", "mid", "far"],
            "results must come back in distance order, not rowid order"
        );
        assert!(
            results[0].distance < 0.001,
            "the self-match must keep its own near-zero distance, got {}",
            results[0].distance
        );
        for pair in results.windows(2) {
            assert!(
                pair[0].distance < pair[1].distance,
                "distances must strictly ascend, or the order is a tie-break: {:?}",
                results.iter().map(|r| r.distance).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn chunk_id_batch_lookup_handles_4096_bound_parameters() {
        let store = VectorStore::open_in_memory(4).unwrap();
        for index in 1..=4096 {
            store
                .conn
                .execute(
                    "INSERT INTO chunk_id_map (rowid, chunk_id) VALUES (?1, ?2)",
                    params![index, format!("chunk-{index}")],
                )
                .unwrap();
        }

        let rowids: Vec<i64> = (1..=4096).collect();
        let mapped = store.chunk_ids_for_rowids(&rowids).unwrap();

        assert_eq!(mapped.len(), 4096);
        assert_eq!(mapped.get(&1).map(String::as_str), Some("chunk-1"));
        assert_eq!(mapped.get(&4096).map(String::as_str), Some("chunk-4096"));
    }

    #[test]
    fn search_reports_missing_chunk_id_mapping() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("missing", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        store
            .conn
            .execute(
                "DELETE FROM chunk_id_map WHERE chunk_id = ?1",
                params!["missing"],
            )
            .unwrap();

        let error = store.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap_err();
        assert!(
            error.to_string().contains("failed to map rowid"),
            "{error:#}"
        );
    }

    #[test]
    fn search_clamps_limit_above_sqlite_vec_knn_cap() {
        let store = VectorStore::open_in_memory(4).unwrap();
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            store.insert(&format!("c{i}"), &v).unwrap();
        }

        // Without clamping, sqlite-vec fails the query outright with
        // "k value in knn query too large".
        let results = store
            .search(&[5.0, 0.0, 0.0, 0.0], MAX_KNN_K + 1)
            .expect("oversized k must be clamped, not rejected");
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn search_respects_limit() {
        let store = VectorStore::open_in_memory(4).unwrap();
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            store.insert(&format!("c{i}"), &v).unwrap();
        }

        let results = store.search(&[5.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn delete_vector() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("c1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert("c2", &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(store.count().unwrap(), 2);

        assert!(store.delete("c1").unwrap());
        assert_eq!(store.count().unwrap(), 1);

        // Deleting non-existent returns false.
        assert!(!store.delete("nonexistent").unwrap());
    }

    #[test]
    fn clear_vectors() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("c1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert("c2", &[5.0, 6.0, 7.0, 8.0]).unwrap();

        store.clear().unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn batch_insert() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("c1", &[1.0, 0.0, 0.0, 0.0]),
            ("c2", &[0.0, 1.0, 0.0, 0.0]),
            ("c3", &[0.0, 0.0, 1.0, 0.0]),
        ];
        store.insert_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 3);
    }

    #[test]
    fn insert_batch_fresh_inserts_and_rejects_duplicates() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("c1", &[1.0, 0.0, 0.0, 0.0]),
            ("c2", &[0.0, 1.0, 0.0, 0.0]),
            ("c3", &[0.0, 0.0, 1.0, 0.0]),
        ];
        store.insert_batch_fresh(&items).unwrap();
        assert_eq!(store.count().unwrap(), 3);

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001);

        // A pre-existing chunk id must fail loudly, not silently remap.
        let dup: Vec<(&str, &[f32])> = vec![("c1", &[0.0, 0.0, 1.0, 0.0])];
        assert!(store.insert_batch_fresh(&dup).is_err());
        assert_eq!(store.count().unwrap(), 3);
    }

    #[test]
    fn higher_dim_vectors_work() {
        // Validate with 4096-dim (Qwen3 production dimensionality).
        let dim = 4096;
        let store = VectorStore::open_in_memory(dim).unwrap();

        let v1 = random_vector(dim, 42);
        let v2 = random_vector(dim, 123);
        let v3 = random_vector(dim, 456);

        store.insert("c1", &v1).unwrap();
        store.insert("c2", &v2).unwrap();
        store.insert("c3", &v3).unwrap();

        assert_eq!(store.count().unwrap(), 3);

        // Self-query should find the same vector.
        let results = store.search(&v1, 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
    }

    #[test]
    fn query_dimension_mismatch_rejected() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("c1", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = store.search(&[1.0, 2.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn insert_same_chunk_preserves_rowid_no_orphans() {
        // Verify that re-inserting a chunk_id updates the vector in-place
        // without creating orphaned rows (the AUTOINCREMENT fix).
        let store = VectorStore::open_in_memory(4).unwrap();

        // Insert initial vector.
        store.insert("c1", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(store.count().unwrap(), 1);

        // Re-insert same chunk_id with a different vector.
        store.insert("c1", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            store.count().unwrap(),
            1,
            "count should still be 1 after re-insert"
        );

        // Search should find the updated vector, not the old one.
        let results = store.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001, "should match updated vector");

        // Old vector should not be a close match.
        let results_old = store.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        // The only result should be c1, and it should have nonzero distance
        // from the old vector since we updated it.
        assert_eq!(results_old.len(), 1);
        assert!(
            results_old[0].distance > 0.5,
            "old vector should not match closely"
        );
    }

    #[test]
    fn batch_insert_same_chunk_no_orphans() {
        let store = VectorStore::open_in_memory(4).unwrap();

        // Insert initial batch.
        let items: Vec<(&str, &[f32])> =
            vec![("c1", &[1.0, 0.0, 0.0, 0.0]), ("c2", &[0.0, 1.0, 0.0, 0.0])];
        store.insert_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 2);

        // Re-insert c1 with updated vector.
        let items2: Vec<(&str, &[f32])> = vec![("c1", &[0.0, 0.0, 1.0, 0.0])];
        store.insert_batch(&items2).unwrap();
        assert_eq!(store.count().unwrap(), 2, "count should still be 2");

        // Verify c1 has the updated vector.
        let results = store.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn orphan_mapping_row_does_not_inflate_count() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("real", &[1.0, 0.0, 0.0, 0.0]).unwrap();

        // Simulate a database written before single inserts became
        // transactional: a mapping row whose vector write never landed.
        store
            .conn
            .execute(
                "INSERT INTO chunk_id_map (chunk_id) VALUES (?1)",
                params!["ghost"],
            )
            .unwrap();

        assert_eq!(
            store.count().unwrap(),
            1,
            "count must report stored vectors, not mapping rows"
        );
        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "real");
    }

    #[test]
    fn batch_failure_rolls_back_earlier_items() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("good-1", &[1.0, 0.0, 0.0, 0.0]),
            ("bad-dim", &[1.0, 0.0]), // dimension mismatch aborts mid-batch
        ];
        assert!(store.insert_batch(&items).is_err());

        // The earlier item's mapping row and vector are both gone: the
        // transaction rolls back when commit never runs.
        assert_eq!(store.count().unwrap(), 0);
        assert!(store.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap().is_empty());
    }

    #[test]
    fn delete_by_file_prefix() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/main.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/main.rs:1", &[0.0, 1.0, 0.0, 0.0]),
            ("src/lib.rs:0", &[0.0, 0.0, 1.0, 0.0]),
        ];
        store.insert_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 3);

        // Delete all vectors for src/main.rs.
        let deleted = store.delete_by_file_prefix("src/main.rs:").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.count().unwrap(), 1);

        // Remaining vector should be src/lib.rs:0.
        let results = store.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "src/lib.rs:0");
    }

    #[test]
    fn delete_by_file_prefix_does_not_treat_wildcards_as_patterns() {
        // `_` matches any character and `%` any sequence in LIKE. A range
        // predicate has no wildcards, but the paths that would have been
        // mis-matched are exactly the ones worth pinning.
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/a_b.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/axb.rs:0", &[0.0, 1.0, 0.0, 0.0]),
            ("src/100%.rs:0", &[0.0, 0.0, 1.0, 0.0]),
            ("src/100pct.rs:0", &[0.0, 0.0, 0.0, 1.0]),
        ];
        store.insert_batch(&items).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/a_b.rs:").unwrap(), 1);
        assert_eq!(store.delete_by_file_prefix("src/100%.rs:").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 2);

        // The literal-looking siblings survive.
        let remaining = store.search(&[0.0, 1.0, 0.0, 0.0], 4).unwrap();
        let ids: Vec<&str> = remaining.iter().map(|r| r.chunk_id.as_str()).collect();
        assert!(ids.contains(&"src/axb.rs:0"), "{ids:?}");
        assert!(ids.contains(&"src/100pct.rs:0"), "{ids:?}");
    }

    #[test]
    fn delete_by_file_prefix_stops_at_the_prefix_boundary() {
        // The upper bound must exclude the next sibling but include every
        // descendant of the prefix, however long.
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/app.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/app.rs:10", &[0.0, 1.0, 0.0, 0.0]),
            ("src/app2.rs:0", &[0.0, 0.0, 1.0, 0.0]),
            ("src/apq.rs:0", &[0.0, 0.0, 0.0, 1.0]),
        ];
        store.insert_batch(&items).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/app.rs:").unwrap(), 2);
        assert_eq!(store.count().unwrap(), 2);
        let remaining = store.search(&[0.0, 0.0, 1.0, 0.0], 4).unwrap();
        let ids: Vec<&str> = remaining.iter().map(|r| r.chunk_id.as_str()).collect();
        assert!(ids.contains(&"src/app2.rs:0"), "{ids:?}");
        assert!(ids.contains(&"src/apq.rs:0"), "{ids:?}");
    }

    #[test]
    fn delete_by_file_prefix_handles_non_ascii_paths() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/café.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/cafz.rs:0", &[0.0, 1.0, 0.0, 0.0]),
        ];
        store.insert_batch(&items).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/café.rs:").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 1);
        let remaining = store.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(remaining[0].chunk_id, "src/cafz.rs:0");
    }

    #[test]
    fn prefix_upper_bound_covers_the_awkward_code_points() {
        assert_eq!(prefix_upper_bound("src/a"), Some("src/b".to_string()));
        assert_eq!(prefix_upper_bound("a:"), Some("a;".to_string()));

        // Successor lands in the surrogate range and must be skipped.
        let below_surrogates = char::from_u32(0xD7FF).unwrap();
        let bound = prefix_upper_bound(&below_surrogates.to_string()).unwrap();
        assert_eq!(bound.chars().next(), char::from_u32(0xE000));

        // A trailing char::MAX has no successor, so the previous character is
        // incremented and the max is dropped.
        let trailing_max = format!("a{}", char::MAX);
        assert_eq!(prefix_upper_bound(&trailing_max), Some("b".to_string()));

        // Nothing sorts above these.
        assert_eq!(prefix_upper_bound(""), None);
        assert_eq!(prefix_upper_bound(&char::MAX.to_string()), None);
    }

    #[test]
    fn prefix_delete_releases_transaction_for_subsequent_writes() {
        // A second connection can write after the prefix delete commits, which
        // confirms that the transaction was committed and released.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let store = VectorStore::open(&path, 4).unwrap();
        store
            .insert_batch(&[("src/a.rs:0", &[1.0, 0.0, 0.0, 0.0][..])])
            .unwrap();

        let other = VectorStore::open(&path, 4).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/a.rs:").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 0);

        // The second connection still works afterwards: the transaction was
        // committed and released, not left open.
        other
            .insert_batch(&[("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..])])
            .unwrap();
        assert_eq!(other.count().unwrap(), 1);
        assert_eq!(other.delete_by_file_prefix("src/a.rs:").unwrap(), 1);
    }

    #[test]
    fn prefix_delete_takes_the_write_lock_before_it_scans() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let store = VectorStore::open(&path, 4).unwrap();
        store
            .insert_batch(&[("src/a.rs:0", &[1.0, 0.0, 0.0, 0.0][..])])
            .unwrap();

        let competitor = VectorStore::open(&path, 4).unwrap();
        competitor
            .conn
            .execute_batch("PRAGMA busy_timeout=0")
            .unwrap();

        let deleted = store
            .delete_by_file_prefix_after_scan("src/a.rs:", || {
                let err = competitor
                    .insert_batch(&[("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..])])
                    .expect_err("the prefix transaction must already hold the write lock");
                assert!(
                    err.chain().any(|cause| matches!(
                        cause.downcast_ref::<rusqlite::Error>(),
                        Some(rusqlite::Error::SqliteFailure(failure, _))
                            if matches!(
                                failure.code,
                                rusqlite::ErrorCode::DatabaseBusy
                                    | rusqlite::ErrorCode::DatabaseLocked
                            )
                    )),
                    "competing write should fail because the database is locked: {err:#}"
                );
            })
            .unwrap();

        assert_eq!(deleted, 1);
        assert_eq!(store.count().unwrap(), 0);
        competitor
            .insert_batch(&[("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..])])
            .unwrap();
        assert_eq!(competitor.count().unwrap(), 1);
    }

    #[test]
    fn prefix_range_seeks_the_index_instead_of_scanning() {
        // The whole point of the range form: `LIKE ... ESCAPE` cannot use the
        // index, so a plan check is what actually guards this.
        let store = VectorStore::open_in_memory(4).unwrap();
        let mut stmt = store
            .conn
            .prepare(&format!(
                "EXPLAIN QUERY PLAN\n                 {PREFIX_RANGE_SQL}"
            ))
            .unwrap();
        let plan: Vec<String> = stmt
            .query_map(params!["a", "b"], |row| row.get::<_, String>(3))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        let plan = plan.join(" | ");
        assert!(
            plan.contains("SEARCH")
                && (plan.contains("USING COVERING INDEX") || plan.contains("USING INDEX")),
            "prefix range must search using an index: {plan}"
        );
    }

    fn assert_search_results_match(
        left: &[VectorSearchResult],
        right: &[VectorSearchResult],
        tolerance: f64,
    ) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.chunk_id, right.chunk_id);
            assert!(
                (left.distance - right.distance).abs() <= tolerance,
                "distance mismatch: left={}, right={}",
                left.distance,
                right.distance
            );
        }
        for pair in left.windows(2) {
            assert!(pair[0].distance <= pair[1].distance);
        }
    }

    fn parity_fixture() -> Vec<(&'static str, &'static [f32])> {
        vec![
            ("a", &[1.0, 0.0, 0.0, 0.0]),
            ("b", &[0.0, 1.0, 0.0, 0.0]),
            ("c", &[-1.0, 0.0, 0.0, 0.0]),
            ("d", &[0.0, 0.0, 1.0, 0.0]),
            ("e", &[0.5, 0.2, 0.1, 0.0]),
        ]
    }

    #[test]
    fn flat_scan_matches_vec0_ids_and_distances() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let flat = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Flat).unwrap();
        flat.insert_batch(&parity_fixture()).unwrap();
        let vec0 = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Vec0).unwrap();

        let flat_results = flat.search(&[0.9, 0.2, 0.1, 0.0], 5).unwrap();
        let vec0_results = vec0.search(&[0.9, 0.2, 0.1, 0.0], 5).unwrap();
        assert_search_results_match(&flat_results, &vec0_results, 1e-5);
    }

    #[test]
    fn flat_incremental_update_preserves_prefix_and_appends_at_row_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let store = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Flat).unwrap();
        store.insert_batch(&parity_fixture()).unwrap();

        let flat_path = dir.path().join(FLAT_FILE_NAME);
        let before = fs::read(&flat_path).unwrap();
        #[cfg(unix)]
        let before_inode = fs::metadata(&flat_path).unwrap().ino();

        let updated = [0.2, 0.3, 0.4, 0.5];
        let appended = [1.5, -0.5, 0.25, 2.0];
        store
            .insert_batch(&[("c", &updated[..]), ("new", &appended[..])])
            .unwrap();

        let after = fs::read(&flat_path).unwrap();
        let changed_offset = vector_byte_offset(3, 4).unwrap() as usize;
        assert_eq!(
            &before[..changed_offset],
            &after[..changed_offset],
            "rows before the first changed rowid must not be rewritten"
        );

        let new_rowid = store.rowids_for_chunk_ids(&["new"])[0];
        let appended_offset = vector_byte_offset(new_rowid, 4).unwrap() as usize;
        let expected_appended: Vec<u8> = appended
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(&after[appended_offset..], expected_appended.as_slice());
        assert_eq!(
            after.len(),
            expected_flat_bytes(4, new_rowid).unwrap() as usize
        );

        #[cfg(unix)]
        assert_eq!(
            before_inode,
            fs::metadata(&flat_path).unwrap().ino(),
            "incremental updates must retain the flat file inode"
        );
    }

    #[test]
    fn flat_incremental_mixed_updates_match_vec0() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let flat = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Flat).unwrap();
        flat.insert_batch(&parity_fixture()).unwrap();
        assert!(flat.delete("b").unwrap());
        assert_eq!(flat.delete_by_file_prefix("d").unwrap(), 1);

        let updated = [0.2, 0.3, 0.4, 0.5];
        let appended = [1.5, -0.5, 0.25, 2.0];
        flat.insert_batch(&[("c", &updated[..]), ("new", &appended[..])])
            .unwrap();

        let vec0 = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Vec0).unwrap();
        for query in [[0.9, 0.2, 0.1, 0.0], [0.0, 0.0, 1.0, 0.0]] {
            let flat_results = flat.search(&query, 5).unwrap();
            let vec0_results = vec0.search(&query, 5).unwrap();
            assert_search_results_match(&flat_results, &vec0_results, 1e-5);
        }
    }

    #[test]
    fn flat_scan_excludes_tombstoned_rows() {
        let store = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Flat).unwrap();
        store
            .insert_batch(&[
                ("near", &[1.0, 0.0, 0.0, 0.0][..]),
                ("far", &[0.0, 1.0, 0.0, 0.0][..]),
            ])
            .unwrap();
        assert!(store.delete("near").unwrap());

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "far");
    }

    #[test]
    fn flat_delete_then_reinsert_same_chunk_id_uses_new_row() {
        let store = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Flat).unwrap();
        store.insert("chunk", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let old_rowid = store.rowids_for_chunk_ids(&["chunk"])[0];
        assert!(store.delete("chunk").unwrap());
        store.insert("chunk", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let new_rowid = store.rowids_for_chunk_ids(&["chunk"])[0];

        assert!(new_rowid > old_rowid);
        let results = store.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "chunk");
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn flat_prefix_delete_then_reinsert_batch_restores_rows() {
        let store = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Flat).unwrap();
        store
            .insert_batch(&[
                ("src/a.rs:0", &[1.0, 0.0, 0.0, 0.0][..]),
                ("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..]),
                ("src/b.rs:0", &[0.0, 0.0, 1.0, 0.0][..]),
            ])
            .unwrap();
        assert_eq!(store.delete_by_file_prefix("src/a.rs:").unwrap(), 2);
        store
            .insert_batch(&[
                ("src/a.rs:2", &[1.0, 1.0, 0.0, 0.0][..]),
                ("src/a.rs:3", &[1.0, 0.0, 1.0, 0.0][..]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 1.0, 0.0, 0.0], 5).unwrap();
        let ids: Vec<_> = results
            .iter()
            .map(|result| result.chunk_id.as_str())
            .collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"src/a.rs:2"));
        assert!(ids.contains(&"src/a.rs:3"));
        assert!(ids.contains(&"src/b.rs:0"));
        assert!(!ids.contains(&"src/a.rs:0"));
        assert!(!ids.contains(&"src/a.rs:1"));
    }

    #[test]
    fn flat_manifest_mismatch_rebuilds_from_vec0() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let store = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Flat).unwrap();
        store.insert_batch(&parity_fixture()).unwrap();
        drop(store);

        fs::write(dir.path().join(MANIFEST_FILE_NAME), b"not-json").unwrap();
        let reopened = VectorStore::open_disk_with_mode(&path, 4, VectorScanMode::Flat).unwrap();
        let results = reopened.search(&[0.9, 0.2, 0.1, 0.0], 5).unwrap();
        assert_eq!(results[0].chunk_id, "a");
        assert!(
            results
                .windows(2)
                .all(|pair| pair[0].distance <= pair[1].distance)
        );
        assert!(dir.path().join(FLAT_FILE_NAME).metadata().unwrap().len() > 0);
    }

    #[test]
    fn flat_and_vec0_in_memory_results_match() {
        let flat = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Flat).unwrap();
        let vec0 = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Vec0).unwrap();
        let fixture = parity_fixture();
        flat.insert_batch(&fixture).unwrap();
        vec0.insert_batch(&fixture).unwrap();

        let flat_results = flat.search(&[0.9, 0.2, 0.1, 0.0], 5).unwrap();
        let vec0_results = vec0.search(&[0.9, 0.2, 0.1, 0.0], 5).unwrap();
        assert_search_results_match(&flat_results, &vec0_results, 1e-5);
    }

    #[test]
    fn flat_scan_clamps_limit_above_knn_cap() {
        let store = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Flat).unwrap();
        for index in 0..10 {
            store
                .insert(&format!("chunk-{index}"), &[index as f32, 0.0, 0.0, 0.0])
                .unwrap();
        }

        let results = store.search(&[5.0, 0.0, 0.0, 0.0], MAX_KNN_K + 1).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn flat_scan_returns_full_depth_above_legacy_knn_cap() {
        // VAL-FILTER-009: flat scanner must honor depth bounded by the actual
        // vector count (values.len() / dim), not the legacy 4096 clamp.
        // This test would have failed when `MAX_KNN_K` was used as a hard flat cap.
        let dim = 4;
        let store = VectorStore::open_in_memory_with_mode(dim, VectorScanMode::Flat).unwrap();
        let count = 5000;
        for index in 0..count {
            store
                .insert(&format!("chunk-{index:04}"), &[index as f32, 0.0, 0.0, 0.0])
                .unwrap();
        }
        assert_eq!(store.count().unwrap(), count as u64);

        let query = vec![2500.0, 0.0, 0.0, 0.0];

        // Requesting the full depth must return the full depth on flat.
        let all = store.search(&query, count).unwrap();
        assert_eq!(
            all.len(),
            count,
            "flat scan must return full depth above legacy 4096 cap"
        );
        assert!(all.len() > MAX_KNN_K);
        // Distances must be sorted.
        assert!(
            all.windows(2)
                .all(|pair| pair[0].distance <= pair[1].distance)
        );

        // Small limit still works.
        let ten = store.search(&query, 10).unwrap();
        assert_eq!(ten.len(), 10);

        // The cap constant itself must remain vec0-specific in the doc comment.
        // Guard against reintroducing the hard clamp by checking the doc still says vec0-only.
        assert_eq!(MAX_KNN_K, 4096);
    }

    #[test]
    fn chunk_ids_for_rowids_batches_above_sql_variable_limit() {
        // VAL-FILTER-010: chunk_ids_for_rowids must handle >999 rowids without
        // "too many SQL variables". The fix batches at 900.
        let store = VectorStore::open_in_memory_with_mode(4, VectorScanMode::Flat).unwrap();
        let count = 2500;
        for index in 0..count {
            store
                .insert(&format!("c{index:04}"), &[index as f32, 0.0, 0.0, 0.0])
                .unwrap();
        }
        let rowids: Vec<i64> = (1..=count).map(|value| value as i64).collect();
        let map = store.chunk_ids_for_rowids(&rowids).unwrap();
        assert_eq!(map.len(), count as usize);
        // Every rowid is present and maps to the expected chunk id.
        assert_eq!(map.get(&1).unwrap(), "c0000");
        assert_eq!(map.get(&901).unwrap(), "c0900");
        assert_eq!(map.get(&2500).unwrap(), "c2499");
        // Re-projecting in caller order must restore the rowid order.
        let reprojected: Vec<String> = rowids.iter().map(|rowid| map[rowid].clone()).collect();
        assert_eq!(reprojected[0], "c0000");
        assert_eq!(reprojected[900], "c0900");
        assert_eq!(reprojected[2499], "c2499");
    }

    #[test]
    fn flat_scan_reaches_low_ranking_chunk_when_fetching_whole_index() {
        // VAL-FILTER-002 analog in unit form: a filtered query whose only
        // matching chunk ranks last must still be reachable when the flat path
        // fetches the whole index. Simulate by inserting 4100+ vectors where
        // the query is close to the first 4099 and far from the last.
        let dim = 4;
        let store = VectorStore::open_in_memory_with_mode(dim, VectorScanMode::Flat).unwrap();
        // 4096 audio-like vectors near [1,0,0,0]
        for index in 0..4096 {
            store
                .insert(
                    &format!("audio:{index:04}"),
                    &[1.0 + index as f32 * 0.0001, 0.0, 0.0, 0.0],
                )
                .unwrap();
        }
        // 12 island vectors far away near [0,1,0,0] — low ranking for audio query
        for index in 0..12 {
            store
                .insert(&format!("video:{index:02}"), &[0.0, 1.0, 0.0, 0.0])
                .unwrap();
        }
        assert_eq!(store.count().unwrap(), 4108);

        // Audio query: island will rank last (distance ~2 vs ~0)
        let audio_query = vec![1.0, 0.0, 0.0, 0.0];
        // Fetching the whole index must include the island chunks
        let all = store.search(&audio_query, 4108).unwrap();
        assert_eq!(all.len(), 4108);
        let island_seen = all
            .iter()
            .any(|result| result.chunk_id.starts_with("video:"));
        assert!(
            island_seen,
            "flat full-depth fetch must include low-ranking island"
        );

        // Fetching only 4096 would miss at least one island chunk when island ranks last
        let capped = store.search(&audio_query, 4096).unwrap();
        assert_eq!(capped.len(), 4096);
        // With 4108 total and 12 island at the tail, top 4096 contains at most 0-? islands.
        // The point is that flat's ability to fetch 4108 is what makes loss impossible.
    }
}
