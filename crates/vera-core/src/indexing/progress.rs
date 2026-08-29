//! Honest indexing progress rendering.
//!
//! While parsing is still in progress, the embedding stage renders an
//! open-ended indicator (count without total) because `cliclack` has no
//! unset-length API. Once `ParsingDone` arrives, the display switches to a
//! fixed, correct total. This module provides the pure state machine that
//! the CLI uses to decide what to render, and is the target of the
//! event-level regression tests.

use super::pipeline::IndexProgress;
use super::update::UpdateProgress;

/// What the embedding indicator should display for a given progress event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedDisplay {
    /// Parsing is still open: show work done without a fixed total and without a
    /// percentage. Example: "Generating embeddings (3,500 chunks so far)".
    Indeterminate { done: usize },
    /// Parsing is complete: show `done/total` against the fixed total and a
    /// monotone percentage.
    Determinate { done: usize, total: usize },
    /// All embeddings for this run are done.
    Done { count: usize },
}

impl EmbedDisplay {
    /// Human message for this display variant.
    pub fn message(&self) -> String {
        match self {
            EmbedDisplay::Indeterminate { done } => {
                format!("Generating embeddings ({} chunks so far)", done)
            }
            EmbedDisplay::Determinate { done, total } => {
                format!("Generating embeddings ({}/{})", done, total)
            }
            EmbedDisplay::Done { count } => format!("Generated {} embeddings", count),
        }
    }

    /// Whether this variant implies a percentage against a known fixed total.
    pub fn shows_percentage(&self) -> bool {
        matches!(self, EmbedDisplay::Determinate { .. })
    }

    /// Fixed total if this is determinate, otherwise None.
    pub fn fixed_total(&self) -> Option<usize> {
        match self {
            EmbedDisplay::Determinate { total, .. } => Some(*total),
            _ => None,
        }
    }
}

/// Pure state machine for honest denominator rendering.
///
/// It observes the `IndexProgress` stream and decides whether the embed
/// indicator is indeterminate or determinate, and what fixed total to use.
#[derive(Debug, Default)]
pub struct HonestProgressTracker {
    parsing_done: Option<usize>,
    last_done: usize,
    last_display: Option<EmbedDisplay>,
}

impl HonestProgressTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one `IndexProgress` event and return the embed display that
    /// should be rendered for it, if any. Non-embedding events return `None`.
    pub fn handle(&mut self, event: &IndexProgress) -> Option<EmbedDisplay> {
        match event {
            IndexProgress::DiscoveryDone { .. } => None,
            IndexProgress::ParsingDone { chunk_count } => {
                self.parsing_done = Some(*chunk_count);
                None
            }
            IndexProgress::EmbeddingProgress { done, .. } => {
                // `done` must be monotonic; callers should ensure this, but we
                // track it for debugging. The total carried by the event is
                // intentionally ignored while parsing is open — it is the
                // per-window parsed count and would imply a recalculating
                // denominator if shown.
                if *done < self.last_done {
                    // Backward movement would be a bug in the pipeline; we
                    // still return a display but note it.
                    tracing::warn!(
                        previous_done = self.last_done,
                        done = *done,
                        "embedding progress moved backward"
                    );
                }
                self.last_done = *done;
                let display = if let Some(fixed) = self.parsing_done {
                    EmbedDisplay::Determinate {
                        done: *done,
                        total: fixed,
                    }
                } else {
                    EmbedDisplay::Indeterminate { done: *done }
                };
                self.last_display = Some(display.clone());
                Some(display)
            }
            IndexProgress::EmbeddingDone { count } => {
                let display = EmbedDisplay::Done { count: *count };
                self.last_display = Some(display.clone());
                Some(display)
            }
            IndexProgress::StorageDone => None,
        }
    }

    /// Whether parsing has completed and the fixed total is known.
    pub fn is_parsing_done(&self) -> bool {
        self.parsing_done.is_some()
    }

    /// Fixed total if parsing is done, otherwise `None`.
    pub fn fixed_total(&self) -> Option<usize> {
        self.parsing_done
    }

    /// Last embed display, if any.
    pub fn last_display(&self) -> Option<&EmbedDisplay> {
        self.last_display.as_ref()
    }
}

/// Equivalent tracker for the update path.
///
/// Update progress has the same honesty contract: no growing denominator
/// presented as a fixed total. The incremental update pipeline parses all
/// changed files before embedding, so `ParsingDone` typically arrives before
/// any `EmbeddingProgress`, but the tracker handles the general case (and
/// preserves the contract if the update pipeline ever becomes windowed).
#[derive(Debug, Default)]
pub struct UpdateProgressTracker {
    parsing_done: Option<usize>,
    last_done: usize,
}

impl UpdateProgressTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&mut self, event: &UpdateProgress) -> Option<EmbedDisplay> {
        match event {
            UpdateProgress::DiscoveryDone { .. } => None,
            UpdateProgress::ClassificationDone { .. } => None,
            UpdateProgress::ParsingDone { chunk_count, .. } => {
                self.parsing_done = Some(*chunk_count);
                None
            }
            UpdateProgress::EmbeddingProgress { done, .. } => {
                if *done < self.last_done {
                    tracing::warn!(
                        previous_done = self.last_done,
                        done = *done,
                        "update embedding progress moved backward"
                    );
                }
                self.last_done = *done;
                let display = if let Some(fixed) = self.parsing_done {
                    EmbedDisplay::Determinate {
                        done: *done,
                        total: fixed,
                    }
                } else {
                    EmbedDisplay::Indeterminate { done: *done }
                };
                Some(display)
            }
            UpdateProgress::EmbeddingDone { count } => Some(EmbedDisplay::Done { count: *count }),
            UpdateProgress::StorageDone => None,
        }
    }

    pub fn is_parsing_done(&self) -> bool {
        self.parsing_done.is_some()
    }

    pub fn fixed_total(&self) -> Option<usize> {
        self.parsing_done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexing::pipeline::IndexProgress;

    fn ind(done: usize) -> EmbedDisplay {
        EmbedDisplay::Indeterminate { done }
    }
    fn det(done: usize, total: usize) -> EmbedDisplay {
        EmbedDisplay::Determinate { done, total }
    }

    #[test]
    fn synthetic_growing_totals_while_parsing_open_renders_no_percentage() {
        let mut tracker = HonestProgressTracker::new();
        // Discovery, then several embedding events with growing per-window totals,
        // all before ParsingDone. Even though the event's total grows, the
        // tracker must stay indeterminate and never show a percentage.
        let events = [
            IndexProgress::DiscoveryDone { file_count: 100 },
            IndexProgress::EmbeddingProgress {
                done: 100,
                total: 100,
            },
            IndexProgress::EmbeddingProgress {
                done: 500,
                total: 500,
            },
            IndexProgress::EmbeddingProgress {
                done: 1000,
                total: 1000,
            },
            IndexProgress::EmbeddingProgress {
                done: 3500,
                total: 3500,
            },
        ];
        for event in &events[1..] {
            let display = tracker.handle(event).unwrap();
            assert_eq!(display, ind(display_message_done(&display)));
            assert!(
                !display.shows_percentage(),
                "must not show percentage while parsing open: {display:?}"
            );
            assert!(
                display.fixed_total().is_none(),
                "must not expose a fixed total while parsing open"
            );
            assert!(
                display.message().contains("chunks so far"),
                "indeterminate message must be open-ended, got: {}",
                display.message()
            );
            assert!(
                !display.message().contains('/'),
                "indeterminate must not contain '/total', got: {}",
                display.message()
            );
        }
        assert!(!tracker.is_parsing_done(), "parsing should still be open");

        // Now parsing completes with the true total.
        let parsing_done = IndexProgress::ParsingDone { chunk_count: 5500 };
        assert!(tracker.handle(&parsing_done).is_none());
        assert_eq!(tracker.fixed_total(), Some(5500));

        // Subsequent embedding must be determinate against the fixed total,
        // regardless of what total the event carries.
        let after = [
            IndexProgress::EmbeddingProgress {
                done: 4000,
                total: 9999,
            },
            IndexProgress::EmbeddingProgress {
                done: 5500,
                total: 1,
            },
            IndexProgress::EmbeddingDone { count: 5500 },
        ];
        for event in &after[..2] {
            let display = tracker.handle(event).unwrap();
            match display {
                EmbedDisplay::Determinate { done, total } => {
                    assert_eq!(total, 5500, "fixed total must be ParsingDone chunk_count");
                    assert!(
                        display.shows_percentage(),
                        "must show percentage after ParsingDone"
                    );
                    assert!(
                        display.message().contains(&format!("{done}/{total}")),
                        "determinate message must contain done/total, got: {}",
                        display.message()
                    );
                    // The event's carried total is ignored; only fixed matters.
                    let _ = done;
                }
                other => panic!("expected determinate after ParsingDone, got {other:?}"),
            }
        }
        let done_display = tracker.handle(&after[2]).unwrap();
        assert_eq!(done_display, EmbedDisplay::Done { count: 5500 });
    }

    fn display_message_done(d: &EmbedDisplay) -> usize {
        match d {
            EmbedDisplay::Indeterminate { done } => *done,
            EmbedDisplay::Determinate { done, .. } => *done,
            EmbedDisplay::Done { count } => *count,
        }
    }

    #[test]
    fn small_single_window_renders_fixed_total_directly() {
        let mut tracker = HonestProgressTracker::new();
        // Small repo: discovery, then ParsingDone for all chunks, then embedding.
        // No indeterminate phase should ever appear.
        let events = [
            IndexProgress::DiscoveryDone { file_count: 10 },
            IndexProgress::ParsingDone { chunk_count: 100 },
            IndexProgress::EmbeddingProgress {
                done: 10,
                total: 100,
            },
            IndexProgress::EmbeddingProgress {
                done: 50,
                total: 100,
            },
            IndexProgress::EmbeddingProgress {
                done: 100,
                total: 100,
            },
        ];
        // First embedding after ParsingDone must be determinate immediately.
        tracker.handle(&events[0]);
        tracker.handle(&events[1]);
        assert!(tracker.is_parsing_done());
        for event in &events[2..] {
            let display = tracker.handle(event).unwrap();
            assert!(
                display.shows_percentage(),
                "small repo must be determinate immediately: {display:?}"
            );
            assert_eq!(display.fixed_total(), Some(100));
            assert!(
                !display.message().contains("so far"),
                "small repo must not flicker through open-ended: {}",
                display.message()
            );
        }
        // Ensure we never went through indeterminate at all.
        // We do this by checking that the first embedding display was determinate.
        let mut fresh = HonestProgressTracker::new();
        fresh.handle(&IndexProgress::DiscoveryDone { file_count: 10 });
        fresh.handle(&IndexProgress::ParsingDone { chunk_count: 100 });
        let first = fresh
            .handle(&IndexProgress::EmbeddingProgress {
                done: 10,
                total: 100,
            })
            .unwrap();
        assert_eq!(first, det(10, 100));
    }

    #[test]
    fn no_backward_percentage_movement_and_fixed_total_never_restated() {
        let mut tracker = HonestProgressTracker::new();
        tracker.handle(&IndexProgress::ParsingDone { chunk_count: 5000 });
        let mut last_done = 0;
        let mut seen_total: Option<usize> = None;
        for done in [1000, 2000, 3500, 5000] {
            let display = tracker
                .handle(&IndexProgress::EmbeddingProgress { done, total: 5000 })
                .unwrap();
            match display {
                EmbedDisplay::Determinate { done: d, total } => {
                    assert!(d >= last_done, "done must be monotonic: {d} < {last_done}");
                    if let Some(prev_total) = seen_total {
                        assert_eq!(
                            prev_total, total,
                            "fixed total must never be restated at a different value"
                        );
                    }
                    seen_total = Some(total);
                    last_done = d;
                }
                _ => panic!("expected determinate"),
            }
        }
        assert_eq!(seen_total, Some(5000));
    }

    #[test]
    fn update_tracker_is_honest() {
        let mut tracker = UpdateProgressTracker::new();
        // Simulate a mid-pipeline embedding before parsing done (hypothetical
        // windowed update); must be indeterminate until ParsingDone.
        let e1 = tracker.handle(&UpdateProgress::EmbeddingProgress {
            done: 50,
            total: 50,
        });
        assert_eq!(e1, Some(ind(50)));
        assert!(!e1.unwrap().shows_percentage());

        tracker.handle(&UpdateProgress::ParsingDone {
            file_count: 5,
            chunk_count: 200,
        });
        let e2 = tracker
            .handle(&UpdateProgress::EmbeddingProgress {
                done: 100,
                total: 200,
            })
            .unwrap();
        assert_eq!(e2, det(100, 200));
        assert!(e2.shows_percentage());
    }
}
