--!nonstrict
-- "/md" launcher provider: up to 30 mediadig results with status rows.

local shared = require("./shared.luau")

local latest = nil

function onQuery(text)
    latest = text
    local q = shared.trim(text)
    if q == "" then
        launcher.setResults(text, { { id = "noop", title = "Type to search memes, videos and documents", glyph = "info-circle" } })
        return
    end
    shared.search(q, 30, 0, false, function(rows)
        if latest == text then
            launcher.setResults(text, rows)
        end
    end)
end

function onActivate(id)
    shared.activate(id)
end
