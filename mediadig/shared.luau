--!nonstrict
-- mediadig launcher helpers shared by the /md provider and the global provider.
-- Queries the mediadig daemon over its UNIX socket with curl; the query reaches sh
-- as positional argument $1, so no shell ever parses it.

local M = {}

local SCRIPT = 'exec curl -sS --max-time 2.5 --unix-socket "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/mediadig/search.sock" --get --data-urlencode "q=$1" --data-urlencode "n=$2" --data-urlencode "min=$3" http://mediadig/search'

-- Global rows stay below app name matches (>= 3000).
local GLOBAL_SCORE_CAP = 2900
local MATCH_LABEL = { words = "caption", meaning = "meaning", both = "caption + meaning" }

function M.trim(s)
    return string.match(s, "^%s*(.-)%s*$")
end

local function info(title, subtitle)
    return { id = "noop", title = title, subtitle = subtitle, glyph = "info-circle" }
end

local function fmtTime(t)
    local s = math.floor(t)
    return string.format("%d:%02d", math.floor(s / 60), s % 60)
end

local function displayPath(path)
    local home = noctalia.getenv("HOME")
    if type(home) == "string" and home ~= "" and string.sub(path, 1, #home + 1) == home .. "/" then
        return "~" .. string.sub(path, #home + 1)
    end
    return path
end

-- The launcher keeps the selection on a row whose id survives a results update. Ids carry
-- the query generation, so a new query's results always start selected at the top.
local generation = 0

local function toRow(r, isGlobal, gen)
    local parts = { displayPath(r.path) }
    if type(r.t) == "number" then
        table.insert(parts, fmtTime(r.t))
    end
    if type(r.page) == "number" then
        table.insert(parts, "p. " .. tostring(math.floor(r.page)))
    end
    local label = MATCH_LABEL[r.match]
    if label then
        table.insert(parts, label)
    end

    local title = r.name
    if r.match ~= "meaning" and type(r.snippet) == "string" and r.snippet ~= "" then
        title = r.snippet
    end

    local code = if r.kind == "video" then "v" else "o"
    local t = if type(r.t) == "number" then r.t else 0
    -- 5 s of lead-in before the matched frame; matches in the first 5 s play from the start.
    local start = string.format("%.1f", math.max(0, t - 5))
    local score = if isGlobal then math.min(GLOBAL_SCORE_CAP, 1000 + 100 * r.score) else 100 * r.score

    local row = {
        id = tostring(gen) .. "\t" .. code .. "\t" .. start .. "\t" .. r.path,
        title = title,
        subtitle = table.concat(parts, " · "),
        score = math.floor(score),
    }
    if type(r.thumb) == "string" then
        row.icon = r.thumb
    else
        row.glyph = "file"
    end
    return row
end

-- deliver(rows) is called exactly once. Global mode delivers {} on every failure.
function M.search(text, n, minScore, isGlobal, deliver)
    generation += 1
    local gen = generation
    local function fail(title, subtitle)
        deliver(if isGlobal then {} else { info(title, subtitle) })
    end

    local argv = { "/bin/sh", "-c", SCRIPT, "mediadig", text, tostring(n), tostring(minScore) }
    local started = noctalia.runAsync(argv, function(res)
        if res.timedOut then
            return fail("mediadig timed out")
        end
        if res.exitCode == 7 then
            return fail("mediadig is not running", "systemctl --user start mediadig")
        end
        if res.exitCode ~= 0 then
            return fail("mediadig error (curl exit " .. tostring(res.exitCode) .. ")")
        end
        local data, err = noctalia.json.decode(res.stdout)
        if type(data) ~= "table" then
            return fail("mediadig error: " .. tostring(err or "bad response"))
        end
        if data.error ~= nil then
            return fail("mediadig error: " .. tostring(data.error))
        end
        local results = if type(data.results) == "table" then data.results else {}
        if #results == 0 then
            return fail("No matches")
        end
        local rows = {}
        for _, r in ipairs(results) do
            table.insert(rows, toRow(r, isGlobal, gen))
        end
        deliver(rows)
    end, 3000)
    if not started then
        fail("Could not run curl")
    end
end

function M.activate(id)
    if id == "noop" then
        return
    end
    local code, start, path = string.match(id, "^%d+\t(%a)\t([^\t]*)\t(.*)$")
    if not code then
        return
    end
    -- No callback: detached (double-fork + setsid), never killed by the launcher.
    if code == "v" then
        noctalia.runAsync({ "mpv", "--start=" .. start, "--", path })
    else
        noctalia.runAsync({ "xdg-open", path })
    end
end

return M
