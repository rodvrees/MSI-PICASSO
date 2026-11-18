// Live log typewriter. The server puts the log text in the dcc.Store "log-text"; this types it into #log-pre one
// character at a time (25 per second), like a film terminal. Text that changed (a progress bar redrawn with \r) is
// cut back to what both versions share and retyped from there. Only a backlog of more than 500 characters (opening the
// page mid-run) types several per tick (backlog / 500). The cursor is solid while typing and blinks when idle.
// The log follows new output unless the reader scrolled up (more than 40 px from the end).
(function () {
    var TICK_MS = 40, CATCH_UP = 500;  // one character per tick, or backlog / 500 when behind
    var target = "", shown = "", timer = null;

    function nodes(pre) {
        var n = pre.__typer;
        if (!n || n.text.parentNode !== pre) {  // first use, or React re-created the element
            n = {text: document.createTextNode(""), cursor: document.createElement("span")};
            n.cursor.className = "cursor";
            n.cursor.textContent = "█";
            pre.textContent = "";
            pre.append(n.text, n.cursor);
            pre.__typer = n;
        }
        return n;
    }

    function render(typing) {
        var pre = document.getElementById("log-pre");
        if (!pre) return false;
        var n = nodes(pre);
        n.text.data = shown;
        n.cursor.classList.toggle("typing", typing);
        if (pre.dataset.follow !== "0") pre.scrollTop = pre.scrollHeight;
        return true;
    }

    function step() {
        var left = target.length - shown.length;
        if (left <= 0) { timer = null; render(false); return; }
        shown = target.slice(0, shown.length + Math.max(1, Math.ceil(left / CATCH_UP)));
        timer = render(true) ? setTimeout(step, TICK_MS) : null;
    }

    window.picassoType = function (text) {
        target = text || "";
        var i = 0, m = Math.min(shown.length, target.length);
        while (i < m && shown.charCodeAt(i) === target.charCodeAt(i)) i++;
        shown = shown.slice(0, i);
        if (!timer) step();
    };

    document.addEventListener("scroll", function (e) {
        var el = e.target;
        if (el && el.id === "log-pre") {
            el.dataset.follow = (el.scrollHeight - el.scrollTop - el.clientHeight < 40) ? "1" : "0";
        }
    }, true);
})();
