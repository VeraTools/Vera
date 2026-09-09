# Agent-Level Vera Benchmark: Flask answer key

## Question 1

### Verified implementation answer
`Flask.__call__` delegates to `wsgi_app`. `wsgi_app` creates one `AppContext` carrying request data, pushes it, calls `full_dispatch_request`, returns the response through WSGI, and always calls `ctx.pop(error)` in `finally`. Pushing sets `_cv_app`, emits `appcontext_pushed`, opens the session, and only then matches the URL. Dispatch emits `request_started`, runs preprocessing and the view, then finalization runs response processing and emits `request_finished`. The active context therefore supplies `current_app`, `g`, `request`, and the session proxy while the request context has request data. A repeated push increments `_push_count` and does not repeat signals, session opening, or matching; cleanup runs only when the matching number of pops reaches zero. `wsgi_app` stores exceptions escaping dispatch in `error` and passes that value to `ctx.pop`.

### Decisive source paths
- `flask/src/flask/app.py:1501-1515`
- `flask/src/flask/app.py:1566-1617`
- `flask/src/flask/ctx.py:416-444`
- `flask/src/flask/app.py:992-1051`
- `flask/src/flask/globals.py:40-62`

### Likely wrong answers
- Flask creates a separate request context and only creates an app context if one is not already active.
- URL matching occurs before session opening because converters cannot use session data.
- `request_finished` is emitted after the context is popped, or request teardown happens before response finalization.

### Scoring rubric
10 points total: 2 points for the `__call__` to `wsgi_app` to `ctx.pop` path; 3 points for context activation, session opening, and URL matching order; 2 points for request dispatch and finalization signal placement; 2 points for repeated-push reference counting; 1 point for passing the escaping exception into pop. Deduct 1 to 3 points for reversing a major ordering edge, and deduct 1 point for each omitted lifecycle stage up to the available points.

### Grader uncertainty notes
The current tree has merged `AppContext` and `RequestContext`; do not penalize an answer that uses the older term "request context" if it describes this combined object and its behavior accurately. The class docstring mentions lazy session loading, but the current `push` implementation calls `_get_session` before matching, so current code takes precedence over that stale wording.

## Question 2

### Verified implementation answer
On the final pop of a request context, Flask first runs request teardown callbacks, then closes the request object, then runs app-context teardown callbacks. Within request teardown, `ctx.request.blueprints` is traversed from the most specific dotted name toward its parents, followed by `None` for app callbacks; each scope's callbacks run in reverse registration order. The request teardown signal is sent after those callbacks. App-context callbacks also run in reverse registration order, followed by the app-context teardown signal. Only after both teardown phases does Flask reset `_cv_app`, then it emits `appcontext_popped`. Teardown errors are collected so later callbacks and the other teardown phase still run; on Python 3.11 or later they are raised as nested `BaseExceptionGroup` values. A handled exception converted to a response inside `full_dispatch_request` normally leaves `wsgi_app`'s `error` as `None`, while an exception escaping dispatch is passed to `ctx.pop` and reaches teardown callbacks. The context pop implementation also collects errors from request close and the popped signal.

### Decisive source paths
- `flask/src/flask/ctx.py:446-504`
- `flask/src/flask/app.py:1420-1479`
- `flask/src/flask/app.py:1566-1617`
- `flask/src/flask/wrappers.py:161-195`
- `flask/tests/test_appctx.py:216-265`

### Likely wrong answers
- App-context teardown runs before request teardown, and the request teardown signal is the final cleanup event.
- Teardown callbacks run in registration order and the first raised exception prevents later callbacks from running.
- An error handled by an error handler is always passed as the original exception to teardown callbacks.

### Scoring rubric
10 points total: 3 points for the request-close, app-teardown, context-reset, and popped-signal ordering; 2 points for dotted blueprint scope traversal; 2 points for reverse registration order; 2 points for error collection and exception-group behavior; 1 point for distinguishing handled and escaping errors. Deduct 2 points for reversing request and app teardown, 1 point for each wrong callback-order claim, and 1 point for failing to mention continued cleanup after a teardown error.

### Grader uncertainty notes
The exact outer exception type for collected teardown failures differs by Python version: the current tests expect nested exception groups on Python 3.11 or later and the first error on older versions. Answers that state this version distinction without reproducing the exact nesting are acceptable.

## Question 3

### Verified implementation answer
The application constructor creates `self.config` through `make_config`. `make_config` chooses `self.root_path` or `self.instance_path` for relative file loading, copies `default_config`, and replaces the default `DEBUG` entry with `get_debug_flag()`. The defaults include `TESTING=False`, `PROPAGATE_EXCEPTIONS=None`, no secret key, a 31-day permanent lifetime, and the session and exception settings shown in `Flask.default_config`. The constructor does not call a file or environment loader; external sources are opt-in method calls. Each loader updates the same mapping, so a later successful update overwrites an earlier value. `from_object`, `from_pyfile`, and `from_mapping` accept only uppercase keys; `from_file` delegates to `from_mapping`; `from_envvar` delegates to `from_pyfile`; and `from_prefixed_env` writes keys after stripping its prefix, including lowercase or mixed-case suffixes because it does not apply `isupper()`.

### Decisive source paths
- `flask/src/flask/sansio/app.py:279-316`
- `flask/src/flask/sansio/app.py:479-493`
- `flask/src/flask/app.py:206-237`
- `flask/src/flask/config.py:94-100`
- `flask/src/flask/config.py:304-321`
- `flask/src/flask/config.py:154-185`

### Likely wrong answers
- Flask automatically loads `.env`, an instance config file, or a package config file in the constructor.
- `DEBUG` always starts as `False` and cannot be sourced from the environment during construction.
- All loaders enforce uppercase keys, including `from_prefixed_env`, or earlier defaults override later file and environment values.

### Scoring rubric
10 points total: 3 points for defaults, `DEBUG`, and root selection; 3 points for last-write-wins update behavior; 2 points for uppercase filtering distinctions; 2 points for recognizing that external loading is explicit. Deduct 2 points for claiming automatic file loading, 1 point for a wrong precedence rule, and 1 point for each incorrect key-filtering claim.

### Grader uncertainty notes
The phrase "precedence" means call order for these mutable loaders, not a hidden global priority among source types. `from_prefixed_env` is intentionally different from the other loaders and can create keys that are not uppercase after prefix removal.

## Question 4

### Verified implementation answer
`from_object` imports a string with `import_string`, iterates `dir(obj)`, and copies only names for which `key.isupper()` is true. `from_pyfile` joins the filename to `root_path`, reads and executes it in a temporary module namespace, then calls `from_object`; its silent mode suppresses `ENOENT`, `EISDIR`, and `ENOTDIR`. `from_file` joins the filename to `root_path`, opens text or binary mode, calls the supplied loader, and passes the result to `from_mapping`; its silent mode suppresses `ENOENT` and `EISDIR`. `from_mapping` combines the mapping and keyword arguments with keywords applied second, then keeps only uppercase keys. `from_envvar` reads the named environment variable, raises `RuntimeError` when it is unset unless silent, and otherwise delegates to `from_pyfile`. `from_prefixed_env` sorts all environment names, selects the computed prefix, strips it, attempts `json.loads` by default, keeps the original string when conversion raises, and writes nested `A__B` paths by creating missing intermediate dictionaries. It does not silently ignore arbitrary loader or execution errors.

### Decisive source paths
- `flask/src/flask/config.py:102-124`
- `flask/src/flask/config.py:126-185`
- `flask/src/flask/config.py:187-216`
- `flask/src/flask/config.py:218-254`
- `flask/src/flask/config.py:256-321`

### Likely wrong answers
- `from_pyfile` imports the file as a normal installed module and silently suppresses every exception from executing it.
- `from_mapping` gives mapping values precedence over keyword arguments and copies lowercase keys.
- `from_prefixed_env` processes environment variables in insertion order, rejects invalid JSON, or treats double underscores as literal key characters.

### Scoring rubric
10 points total: 3 points for object and Python-file loading; 3 points for file, mapping, and environment-variable behavior; 3 points for prefixed-environment parsing; 1 point for the differing silent-error sets. Deduct 1 point for each wrong loader delegation, 1 point for reversing keyword precedence, and 1 to 2 points for missing nested-environment behavior.

### Grader uncertainty notes
The exact `from_pyfile` and `from_file` silent error sets differ. Do not penalize an answer that groups them as "missing-file errors" if the answer otherwise distinguishes that `from_pyfile` also handles `ENOTDIR`; do penalize a claim that all `OSError` values are suppressed.

## Question 5

### Verified implementation answer
An app route decorator calls `Scaffold.route`, whose decorator calls the app's `add_url_rule` immediately. The app implementation chooses the function name when no endpoint is supplied, normalizes methods, adds automatic `OPTIONS` when configured, creates a `Rule`, adds it to `url_map`, and stores the view in `view_functions`, rejecting a different function for an existing endpoint. A blueprint route instead records a deferred callback. At registration, the blueprint computes its dotted registration name, creates `BlueprintSetupState`, merges previously registered callback data, and invokes deferred callbacks. `BlueprintSetupState.add_url_rule` joins the registration prefix with the route, combines URL defaults, and calls the app's `add_url_rule` with an endpoint formed from `name_prefix`, blueprint name, and endpoint. Nested blueprints combine URL prefixes and dotted names. Registering the same effective name twice raises `ValueError`; registering the same blueprint under a different name is allowed, with `record_once` callbacks only applying on the first registration of that blueprint object.

### Decisive source paths
- `flask/src/flask/sansio/scaffold.py:340-365`
- `flask/src/flask/sansio/app.py:601-658`
- `flask/src/flask/sansio/blueprints.py:34-116`
- `flask/src/flask/sansio/blueprints.py:224-253`
- `flask/src/flask/sansio/blueprints.py:273-335`
- `flask/src/flask/sansio/blueprints.py:349-410`

### Likely wrong answers
- A blueprint owns a separate URL map and dispatches its route without copying anything to the app.
- A blueprint endpoint remains only the view function name, and `url_prefix` affects incoming requests but not URL building.
- The same blueprint can be registered repeatedly with the same name and all deferred callbacks always run again.

### Scoring rubric
10 points total: 3 points for the direct app route path; 3 points for deferred blueprint registration and setup-state rule merging; 2 points for dotted endpoints, prefixes, and defaults; 2 points for repeated-registration and collision behavior. Deduct 2 points for treating blueprints as independently dispatched apps, 1 point for omitting endpoint prefixing, and 1 point for each wrong repeated-registration claim.

### Grader uncertainty notes
Answers may describe the endpoint as `name.endpoint` or as the fully dotted `name_prefix.name.endpoint`; both are acceptable if they explain that the resulting endpoint is registered in the app and that nested names are dotted. The route method defaults and automatic `OPTIONS` details are only required when the answer discusses the app rule implementation.

## Question 6

### Verified implementation answer
Methods inherited from `Scaffold` register app or blueprint-local callbacks under the local `None` key. When a blueprint is merged, its local `None` key is rewritten to the registered blueprint name, while already scoped keys receive the registered name as a prefix. `before_app_request`, `after_app_request`, and `teardown_app_request` use `record_once` to append directly to the app's `None` lists, so they apply to every request. `app_errorhandler` similarly records an app-level error handler. For preprocessing, Flask searches `(None, *reversed(req.blueprints))`, so app callbacks run before the parent and most-specific blueprint callbacks. For after processing and request teardown, it searches `req.blueprints` followed by `None`, and reverses each scope's registration list, so the most-specific blueprint scope is processed before its parents and the app scope. Error lookup searches blueprint names from most specific to parent to app, tries the HTTP code bucket before the class bucket, and walks the exception class MRO within each bucket. Thus a blueprint code handler beats an app code handler, which beats a blueprint class handler, which beats an app class handler; a local blueprint handler is only considered for that blueprint's request.

### Decisive source paths
- `flask/src/flask/sansio/scaffold.py:459-555`
- `flask/src/flask/sansio/blueprints.py:379-410`
- `flask/src/flask/sansio/blueprints.py:613-692`
- `flask/src/flask/wrappers.py:161-195`
- `flask/src/flask/app.py:1366-1392`
- `flask/src/flask/app.py:1407-1416`
- `flask/src/flask/sansio/app.py:865-888`

### Likely wrong answers
- A blueprint's ordinary `before_request` or `errorhandler` is copied into the app-wide `None` scope and applies to every route.
- App callbacks run after blueprint callbacks for preprocessing, or after callbacks run app first and blueprint last.
- Exception class handlers are always checked before HTTP status-code handlers, or app handlers always beat blueprint handlers.

### Scoring rubric
10 points total: 3 points for local versus app-wide registration; 2 points for preprocessing order; 2 points for after and teardown order; 3 points for code, scope, and MRO error-handler precedence. Deduct 1 to 2 points for each reversed callback phase, 2 points for reversing code versus class lookup, and 1 point for treating local handlers as global.

### Grader uncertainty notes
For a nested request, `req.blueprints` is most-specific first because `_split_blueprint_path` starts with the full dotted name. Answers that express the same order as "app, outer, inner" for preprocessing and "inner, outer, app" for after/teardown receive full credit even if they do not name the tuple construction.

## Question 7

### Verified implementation answer
The signed-cookie implementation uses `URLSafeTimedSerializer`. If there is no `secret_key`, `get_signing_serializer` returns `None`. Otherwise it builds a key list from configured `SECRET_KEY_FALLBACKS` followed by the current key, uses salt `cookie-session`, HMAC key derivation, the lazy SHA-1 digest, and `TaggedJSONSerializer`. Saving calls `dumps(dict(session))`; loading calls `loads` with `max_age` equal to the configured permanent-session lifetime in seconds. The tagged serializer recursively converts supported values and supports dicts, tuples, bytes, `Markup`, UUIDs, and datetimes. A missing cookie creates an empty `SecureCookieSession`. A tampered or otherwise invalid cookie raises an `itsdangerous` `BadSignature`, which is caught and replaced with a new empty session. An expired timed signature raises `SignatureExpired`, which is a `BadSignature` subclass in itsdangerous, so it follows the same catch and also becomes a new empty session. Neither condition is surfaced by this implementation as a request exception.

### Decisive source paths
- `flask/src/flask/sessions.py:273-335`
- `flask/src/flask/sessions.py:284-321`
- `flask/src/flask/sessions.py:323-335`
- `flask/src/flask/sessions.py:337-374`
- `flask/src/flask/json/tag.py:219-327`

### Likely wrong answers
- Flask stores the session as unsigned JSON and validates it only when the view accesses `session`.
- A bad signature causes a 400 or 500 response, while an expired signature is silently accepted as valid data.
- Only the current secret key is accepted, and fallback keys are tried after the current key or are unrelated to cookie loading.

### Scoring rubric
10 points total: 3 points for serializer construction and key ordering; 2 points for signed timed loading and max-age; 2 points for tagged serialization and supported types; 3 points for identical empty-session recovery from invalid and expired signatures. Deduct 2 points for missing the `BadSignature` catch, 2 points for treating expiration as an application error, and 1 point for each wrong key or serializer detail.

### Grader uncertainty notes
The Flask source imports and catches `BadSignature` rather than naming `SignatureExpired`. Full credit requires recognizing the dependency relationship that makes expired timed signatures enter that handler. Answers that say "invalid timestamp/signature errors" and clearly describe the same empty-session recovery are acceptable.

## Question 8

### Verified implementation answer
`AppContext._get_session` calls `open_session` once and, when it returns `None`, replaces the result with `make_null_session`. The default `NullSession` is readable but all mutating mapping operations raise a `RuntimeError` explaining that a secret key is missing. `process_response` skips `save_session` for a null session. Accessing the session through `ctx.session` sets `session.accessed`, and the signed-cookie saver adds `Vary: Cookie` when accessed. If the session is empty and modified, it deletes the cookie and adds the vary header; if empty and unmodified, it does nothing. A non-empty session is saved only when modified or when it is permanent and `SESSION_REFRESH_EACH_REQUEST` is true. Permanent sessions receive an expiration based on `PERMANENT_SESSION_LIFETIME`; non-permanent sessions do not. Response processing runs after after-request callbacks and before request teardown, so session saving occurs before teardown callbacks and the request is closed.

### Decisive source paths
- `flask/src/flask/ctx.py:381-403`
- `flask/src/flask/sessions.py:83-97`
- `flask/src/flask/sessions.py:150-166`
- `flask/src/flask/sessions.py:223-247`
- `flask/src/flask/app.py:1394-1418`
- `flask/src/flask/sessions.py:337-385`

### Likely wrong answers
- A missing secret key makes every request fail while opening the session, and the null session is still serialized on the response.
- Reading a session always sets a cookie, even when the session is empty and unmodified.
- Session saving happens after request teardown, or permanent sessions are saved only when modified.

### Scoring rubric
10 points total: 3 points for null-session creation, mutation failure, and save skipping; 3 points for accessed, empty, and cookie-header behavior; 2 points for modified, permanent, and refresh decisions; 2 points for placement before teardown. Deduct 2 points for claiming null sessions are saved, 1 point for each wrong cookie condition, and 1 point for reversing response and teardown timing.

### Grader uncertainty notes
The session interface contract says `save_session` is skipped for a null session, and the actual skip is in `process_response`. Answers that attribute the skip to either layer receive credit if they identify both the contract and the call-site behavior accurately.

## Question 9

### Verified implementation answer
Flask creates a Blinker `Namespace` and defines template, request, app-context, exception, and flash signals in `signals.py`. The request path emits `request_started` before preprocessing, `request_finished` after response processing and session saving, `request_tearing_down` after request teardown callbacks, and `got_request_exception` from `handle_exception`. Context push emits `appcontext_pushed`; context pop runs app teardown, resets the context variable, and then emits `appcontext_popped`; app teardown emits `appcontext_tearing_down` after its callbacks. Template rendering emits `before_render_template` before rendering and `template_rendered` after rendering, while streaming sends the latter when generation completes. `flash` updates the session and emits `message_flashed` with message and category. These source files define signals and send them, but do not connect built-in receivers, so there are no default Flask subscribers in the current source. Applications or extensions observe them by calling `.connect(receiver, app)` or equivalent; receivers receive the sender and the signal keyword arguments, such as `response`, `exception`, `exc`, `template`, `context`, `message`, or `category`.

### Decisive source paths
- `flask/src/flask/signals.py:1-16`
- `flask/src/flask/ctx.py:428-444`
- `flask/src/flask/ctx.py:486-504`
- `flask/src/flask/app.py:1012-1019`
- `flask/src/flask/app.py:1039-1044`
- `flask/src/flask/app.py:1420-1479`
- `flask/src/flask/app.py:925-926`
- `flask/src/flask/templating.py:123-178`
- `flask/src/flask/helpers.py:348-357`
- `flask/tests/test_signals.py:50-92`

### Likely wrong answers
- Flask ships default receivers that log every request signal or automatically enable signal handling only in debug mode.
- `request_finished` is emitted before after-request callbacks and session saving, or `request_tearing_down` is emitted before teardown callbacks.
- `message_flashed` is emitted when flashed messages are read rather than when `flash` updates the session.

### Scoring rubric
10 points total: 3 points for identifying signal definitions and request/context emission points; 2 points for template and flash emission; 3 points for ordering relative to callbacks and cleanup; 2 points for the absence of built-in subscribers and receiver argument behavior. Deduct 1 to 2 points for each major signal-order reversal, 2 points for claiming default subscribers, and 1 point for omitting sender versus keyword arguments.

### Grader uncertainty notes
"No default subscribers" means no receiver registrations in the Flask source itself. Blinker may have its own dispatch behavior, and tests or third-party extensions may connect receivers; those are not Flask default subscribers. A receiver connected to an app-specific sender should be described as observing only that sender.

## Question 10

### Verified implementation answer
`full_dispatch_request` catches exceptions from request-start, preprocessing, routing dispatch, and the view and passes them to `handle_user_exception`. An `HTTPException` that is not trapped goes to `handle_http_exception`; routing exceptions are returned unchanged for Werkzeug's routing behavior, and other HTTP exceptions use code-first and MRO-based handler lookup, falling back to the exception response itself. A generic exception with no handler is re-raised from `handle_user_exception`, reaches the outer `wsgi_app` handler, and enters `handle_exception`. `handle_exception` always emits `got_request_exception`; if `PROPAGATE_EXCEPTIONS` is unset it derives propagation from `testing or debug`, and if propagation is true it re-raises. Otherwise it logs the original exception, wraps it in `InternalServerError(original_exception=e)`, applies a 500 handler if present, and safely finalizes that response. `TRAP_HTTP_EXCEPTIONS` traps all HTTP exceptions, while unset `TRAP_BAD_REQUEST_ERRORS` in debug mode traps `BadRequestKeyError`; trapped exceptions continue through ordinary exception handling. The proxies are `LocalProxy` objects backed by the `_cv_app` context variable. `current_app` resolves the active context's `app`; `request` resolves its `request` property. An app-only context can resolve `current_app` but its request property and `request` proxy fail; with no context both proxies fail with their configured runtime messages.

### Decisive source paths
- `flask/src/flask/app.py:830-895`
- `flask/src/flask/app.py:897-948`
- `flask/src/flask/app.py:992-1019`
- `flask/src/flask/app.py:1566-1617`
- `flask/src/flask/sansio/app.py:865-918`
- `flask/src/flask/globals.py:33-62`
- `flask/src/flask/ctx.py:370-403`
- `flask/src/flask/app.py:1481-1515`

### Likely wrong answers
- Every exception, including an unhandled `HTTPException`, is always converted by `handle_exception` into a new 500 response.
- `DEBUG` only changes the response display and never affects propagation; `TESTING` affects extensions but not exception propagation when the config is unset.
- `current_app` and `request` are ordinary process-global objects, and outside a context they return `None` or a stale previous request.

### Scoring rubric
10 points total: 3 points for the separate HTTP and generic exception paths; 3 points for propagation, trapping, signaling, and 500 wrapping; 2 points for debug/testing behavior; 2 points for proxy resolution in app-only, request, and unbound states. Deduct 2 points for treating all HTTP exceptions as generic 500s, 1 point for each wrong propagation setting, and 2 points for claiming proxies are globals or silently return `None`.

### Grader uncertainty notes
An HTTP exception can still be handled by a registered generic `Exception` handler when it is trapped, because trapping prevents the normal HTTP-exception branch. Answers that focus only on the default untrapped path are incomplete but should receive the points for that path. The exact `LocalProxy` exception text need not be quoted if the answer identifies the correct application-versus-request context distinction.
