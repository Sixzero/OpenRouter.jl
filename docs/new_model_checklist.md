# Új modell checklist (pl. claude-fable-5.2 → mindenhol)

Egy helyen, hogy ne kelljen újra kitalálni. Mindig: `rg -n "<előző-verzió-id>"` (pl. `fable-5\.1`) a `packages/`, `frontend/src`, `agent/src` alatt — ami ott hardcode-olt, azt cseréld.

1. **CLIProxyAPI** (`~/repo/awesome/CLIProxyAPI`): `/v1/models` listázza-e (remote catalog, 3h refresh). Ha a kérés `Claude Code X does not support this model; version Y or newer is required` → `claude-header-defaults.user-agent` bump a `config.yaml`-ban **lokálisan (`~/cliproxyapi`) és szerveren (`ssh todoforai`, `/root/cliproxyapi`)**, majd restart (hot-reload nem megbízható), tesztkérés. Lásd FORK_NOTES.md „Claude Code version we impersonate”.
2. **Export**: `cd ~/repo/todoforai/frontend && bun run update:models` → `src/assets/models_data.json`. Diffeld az id-ket (added/removed), a szemetet `scripts/export_models_json.jl` `SUPERSEDED_MODELS`/`EXOTIC_MODELS`-be.
3. **packages** (`~/repo/todoforai/packages`):
   - `shared-fbe/src/modelFamilies.ts` — család `current`
   - `shared-fer/src/modelMentionAliases.ts` — `@fable`/`@opus`/… alias
   - `shared-fer/src/recommendedModels.ts`, `shared-fer/src/modelDescriptions.ts`
   - `shared-fbe/src/thinkingLevels.ts` (Claude 5 család pattern lefedi; csak ha eltér), `shared-fbe/src/model_promos.json` (`family` regex általában lefedi)
4. **frontend**: `src/components/main/DiscountedModelsSection.tsx` promókártyák.
5. **Deploy**: packages `main` push (a frontend CI a packages default branch-ét checkoutolja), frontend `main`→`prod` **csak FF**. Backend a `models_data.json`-t cache-eli (`/api/v1/models`, mobil/CLI) → csak a következő backend FF deploykor frissül; agent és backend soha ne együtt.
