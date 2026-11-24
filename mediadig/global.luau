--!nonstrict
-- Global launcher provider (no prefix): at most 6 strong mediadig results, silent on errors.

local shared = require("./shared.luau")

local latest = nil

function onQuery(text)
    latest = text
    local q = shared.trim(text)
    if #q < 3 then
        launcher.setResults(text, {})
        return
    end
    shared.search(q, 6, 3.0, true, function(rows)
        if latest == text then
            launcher.setResults(text, rows)
        end
    end)
end

function onActivate(id)
    shared.activate(id)
end
