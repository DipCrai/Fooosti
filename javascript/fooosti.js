// Fooosti UI helpers: aspect-ratio preview + per-device settings persistence
var FOOOSTI_KEY = 'fooosti_settings';
var FOOOSTI_TRACKED = [
    'aspect_ratios_selection', 'performance_selection', 'style_selections',
    'image_number', 'input_image_checkbox', 'enhance_checkbox', 'advanced_checkbox'
];

function fooosti_elem(id) {
    var root = gradioApp();
    return root.getElementById ? root.getElementById(id) : root.querySelector('#' + id);
}

function fooosti_ratio_from_label(x) {
    if (!x) return null;
    var m = String(x).match(/(\d+)[x×*](\d+)/);
    if (m && parseInt(m[2], 10) > 0) return parseFloat(m[1]) / parseFloat(m[2]);
    return null;
}

function set_preview_ratio(x) {
    var r = fooosti_ratio_from_label(x);
    if (r) {
        (gradioApp() || document.documentElement).style.setProperty('--fooosti-aspect', r);
    }
}

function fooosti_style_value(input) {
    var lab = input.closest('label');
    var span = lab ? lab.querySelector('span') : null;
    return span ? span.textContent.trim() : input.value;
}

// --- generic input reader/setter over an element's <input>s ---
function fooosti_read_inputs(id, selector, valueOf) {
    var el = fooosti_elem(id);
    if (!el) return undefined;
    var out = [];
    var inputs = el.querySelectorAll(selector);
    for (var i = 0; i < inputs.length; i++) {
        if (inputs[i].checked) out.push(valueOf ? valueOf(inputs[i]) : inputs[i].value);
    }
    return out;
}

function fooosti_set_inputs(id, selector, matches, commit) {
    // matches(input) -> desired state or undefined to leave the input alone;
    // commit(input, want) -> the value to assign (checked/value)
    var el = fooosti_elem(id);
    if (!el) return false;
    var inputs = el.querySelectorAll(selector);
    var changed = [];
    for (var i = 0; i < inputs.length; i++) {
        var want = matches(inputs[i]);
        if (want === undefined) continue;
        var current = inputs[i].checked !== undefined ? inputs[i].checked : String(inputs[i].value);
        var target = commit(inputs[i], want);
        if (current === target) continue;
        if (inputs[i].checked !== undefined) inputs[i].checked = target;
        else inputs[i].value = target;
        changed.push(inputs[i]);
    }
    for (var i = 0; i < changed.length; i++) {
        changed[i].dispatchEvent(new Event('change', {bubbles: true}));
    }
    return changed.length > 0;
}

function fooosti_set_radio(id, val) {
    if (val === undefined || val === null) return false;
    return fooosti_set_inputs(
        id, 'input[type=radio]',
        function (input) { return input.value === String(val) ? true : undefined; },
        function () { return true; }
    );
}

function fooosti_set_checkboxes(id, values) {
    if (!values) return false;
    var want = {};
    for (var i = 0; i < values.length; i++) want[values[i]] = true;
    return fooosti_set_inputs(
        id, 'input[type=checkbox]',
        function (input) { return !!want[fooosti_style_value(input)]; },
        function (input, want) { return want; }
    );
}

function fooosti_set_checkbox(id, val) {
    if (val === undefined || val === null) return false;
    return fooosti_set_inputs(
        id, 'input[type=checkbox]',
        function () { return !!val; },
        function () { return true; }
    );
}

function fooosti_set_range(id, val) {
    if (val === undefined || val === null) return false;
    var el = fooosti_elem(id);
    if (!el) return false;
    var r = el.querySelector('input[type=range]');
    if (r && String(r.value) !== String(val)) {
        r.value = val;
        r.dispatchEvent(new Event('input', {bubbles: true}));
        r.dispatchEvent(new Event('change', {bubbles: true}));
        return true;
    }
    return false;
}

function fooosti_read_value(id, read) {
    var el = fooosti_elem(id);
    if (!el) return undefined;
    var input = el.querySelector(read);
    return input ? input.value : undefined;
}

function fooosti_save() {
    var s = {
        aspect: fooosti_read_value('aspect_ratios_selection', 'input[type=radio]:checked'),
        performance: fooosti_read_value('performance_selection', 'input[type=radio]:checked'),
        styles: fooosti_read_inputs('style_selections', 'input[type=checkbox]', fooosti_style_value),
        imageNumber: fooosti_read_value('image_number', 'input[type=range]'),
        inputImage: fooosti_read_inputs('input_image_checkbox', 'input[type=checkbox]').length > 0,
        enhance: fooosti_read_inputs('enhance_checkbox', 'input[type=checkbox]').length > 0,
        advanced: fooosti_read_inputs('advanced_checkbox', 'input[type=checkbox]').length > 0
    };
    try { localStorage.setItem(FOOOSTI_KEY, JSON.stringify(s)); } catch (e) { console.warn('Fooosti: settings not persisted', e); }
}

function fooosti_restore() {
    var raw;
    try { raw = localStorage.getItem(FOOOSTI_KEY); } catch (e) { console.warn('Fooosti: settings not readable', e); return false; }
    if (!raw) return false;
    var s;
    try { s = JSON.parse(raw); } catch (e) { console.warn('Fooosti: corrupted settings, ignoring', e); return false; }
    var changed = false;
    changed = fooosti_set_radio('aspect_ratios_selection', s.aspect) || changed;
    changed = fooosti_set_radio('performance_selection', s.performance) || changed;
    var styles = s.styles;
    if (styles && styles.indexOf('on') >= 0) styles = undefined;
    changed = fooosti_set_checkboxes('style_selections', styles) || changed;
    changed = fooosti_set_range('image_number', s.imageNumber) || changed;
    changed = fooosti_set_checkbox('input_image_checkbox', s.inputImage) || changed;
    changed = fooosti_set_checkbox('enhance_checkbox', s.enhance) || changed;
    changed = fooosti_set_checkbox('advanced_checkbox', s.advanced) || changed;
    return changed;
}

function fooosti_host_of(t) {
    if (!t || !t.closest) return null;
    var host = t.closest('[id]');
    if (!host) return null;
    for (var i = 0; i < FOOOSTI_TRACKED.length; i++) {
        if (host.id === FOOOSTI_TRACKED[i]) return host.id;
    }
    return null;
}

function fooosti_bind() {
    var root = gradioApp();
    if (root.__fooostiBound) return;
    root.__fooostiBound = true;
    root.addEventListener('change', function (e) {
        if (fooosti_host_of(e.target)) fooosti_save();
    }, true);
    root.addEventListener('input', function (e) {
        if (fooosti_host_of(e.target)) fooosti_save();
    }, true);
}

// hide the app until settings are restored, then reveal once the resulting
// gradio re-render has settled (avoids a flash of un-restored UI)
var fooostiRestored = false;
var fooostiPending = false;
var fooostiDispatchTime = 0;
var fooostiLastMutation = 0;

function fooosti_reveal() {
    document.documentElement.classList.remove('fooosti_loading');
}

function fooosti_poll() {
    if (!fooostiPending) return;
    var now = Date.now();
    // reveal once mutations from our restore have stopped for 250ms, or after
    // 800ms if no mutation was ever observed (nothing re-rendered)
    if ((now - fooostiLastMutation >= 250) || (fooostiLastMutation === fooostiDispatchTime && now - fooostiDispatchTime >= 800)) {
        fooostiPending = false;
        fooosti_reveal();
        return;
    }
    window.setTimeout(fooosti_poll, 150);
}

function fooosti_try_restore() {
    if (fooostiRestored) return;
    if (fooosti_elem('aspect_ratios_selection') && fooosti_elem('style_selections')) {
        fooostiRestored = true;
        var changed = fooosti_restore();
        fooosti_bind();
        if (!changed) {
            fooosti_reveal();
        } else {
            fooostiPending = true;
            fooostiDispatchTime = Date.now();
            fooostiLastMutation = Date.now();
            window.setTimeout(fooosti_poll, 200);
        }
    }
}

document.documentElement.classList.add('fooosti_loading');
document.addEventListener('DOMContentLoaded', function () {
    fooosti_try_restore();
    try {
        var root = gradioApp() || document;
        new MutationObserver(function () {
            fooostiLastMutation = Date.now();
            fooosti_try_restore();
        }).observe(root, {childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class']});
    } catch (e) {
        window.setInterval(fooosti_try_restore, 500);
    }
});
if (typeof onUiUpdate === 'function') {
    onUiUpdate(function () { fooosti_try_restore(); });
}
