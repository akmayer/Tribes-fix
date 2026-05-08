# Fog-of-War Implementation Fix Guide

## Quick Reference

**File to fix:** `src/players/PythonBridge.java`

**Severity:** 🔴 CRITICAL (blocks model training)

**Time estimate:** 30-45 minutes

**Methods affected:** 4
- `buildPayload()` (line 54)
- `serializeTribes()` (line 95)
- `serializeTribeUnits()` (line 150)
- `serializeCities()` (line 175)

---

## The Core Issue

**Current behavior:**
```
PythonBridge sends FULL board state including ALL enemy units and cities
→ NN sees perfect information
→ NN learns perfect-information strategy
→ NN fails when deployed with actual fog-of-war
```

**Required behavior:**
```
PythonBridge sends:
  - Own tribe: ALL units and cities (no FOW for self)
  - Enemy tribes: ONLY units/cities in observability grid (FOW applied)
→ NN sees what actual player sees
→ NN learns FOW-aware strategy
→ NN works correctly in real gameplay
```

---

## Implementation Steps

### Step 1: Update `buildPayload()` method

**Location:** Line 54 in `src/players/PythonBridge.java`

**Current code:**
```java
private static JSONObject buildPayload(GameState gs, ArrayList<Action> allActions) {
    JSONObject payload = new JSONObject();
    Board board = gs.getBoard();

    payload.put("schema_version", 1);
    payload.put("tick", gs.getTick());
    payload.put("active_tribe_id", gs.getActiveTribeID());
    payload.put("game_mode", gs.getGameMode().name());
    payload.put("is_game_over", gs.isGameOver());
    payload.put("available_action_count", allActions.size());
    payload.put("available_actions", serializeActions(allActions, gs));
    payload.put("board", serializeBoard(board));
    payload.put("tribes", serializeTribes(board));     // <-- NEEDS CHANGE

    return payload;
}
```

**Updated code:**
```java
private static JSONObject buildPayload(GameState gs, ArrayList<Action> allActions) {
    JSONObject payload = new JSONObject();
    Board board = gs.getBoard();
    
    // GET OBSERVABILITY GRID FOR FOW FILTERING
    Tribe activeTribe = gs.getActiveTribe();
    boolean[][] obsGrid = activeTribe.getObsGrid();
    int activeTribeID = gs.getActiveTribeID();

    payload.put("schema_version", 1);
    payload.put("tick", gs.getTick());
    payload.put("active_tribe_id", activeTribeID);
    payload.put("game_mode", gs.getGameMode().name());
    payload.put("is_game_over", gs.isGameOver());
    payload.put("available_action_count", allActions.size());
    payload.put("available_actions", serializeActions(allActions, gs));
    payload.put("board", serializeBoard(board));
    payload.put("tribes", serializeTribes(board, obsGrid, activeTribeID));  // <-- PASS FOW INFO

    return payload;
}
```

**Changes:**
- Get active tribe from `gs.getActiveTribe()`
- Extract `obsGrid` (boolean array of what's visible)
- Extract `activeTribeID`
- Pass both to `serializeTribes()`

---

### Step 2: Update `serializeTribes()` signature

**Location:** Line 95 in `src/players/PythonBridge.java`

**Current signature:**
```java
private static JSONArray serializeTribes(Board board) {
```

**New signature:**
```java
private static JSONArray serializeTribes(Board board, boolean[][] obsGrid, int activeTribeID) {
```

**Changes:**
- Add `obsGrid` parameter (what's visible)
- Add `activeTribeID` parameter (which tribe is active)
- Pass these to helper methods

**Inside the method, update calls:**

Current:
```java
t.put("cities", serializeCities(board, tribe));
t.put("units", serializeTribeUnits(board, tribe));
```

Updated:
```java
t.put("cities", serializeCities(board, tribe, obsGrid, activeTribeID));
t.put("units", serializeTribeUnits(board, tribe, obsGrid, activeTribeID));
```

---

### Step 3: Update `serializeTribeUnits()` method

**Location:** Line 150 in `src/players/PythonBridge.java`

**Current signature and start:**
```java
private static JSONArray serializeTribeUnits(Board board, Tribe tribe) {
    JSONArray units = new JSONArray();
    Set<Integer> seen = new HashSet<>();

    for (Integer cityId : tribe.getCitiesID()) {
        City city = (City) board.getActor(cityId);
        if (city == null) {
            continue;
        }
        for (Integer unitId : city.getUnitsID()) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit != null) {
                units.put(serializeUnit(unit));  // <-- NO FOW CHECK
            }
        }
    }
    // ... rest of method
}
```

**New signature and implementation:**
```java
private static JSONArray serializeTribeUnits(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
    JSONArray units = new JSONArray();
    Set<Integer> seen = new HashSet<>();

    // If this is the ACTIVE tribe, include all units (no FOW for self)
    if (tribe.getTribeId() == activeTribeID) {
        for (Integer cityId : tribe.getCitiesID()) {
            City city = (City) board.getActor(cityId);
            if (city == null) {
                continue;
            }
            for (Integer unitId : city.getUnitsID()) {
                if (unitId == null || seen.contains(unitId)) {
                    continue;
                }
                seen.add(unitId);
                Unit unit = (Unit) board.getActor(unitId);
                if (unit != null) {
                    units.put(serializeUnit(unit));
                }
            }
        }
        // Include extra units
        for (Integer unitId : tribe.getExtraUnits()) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit != null) {
                units.put(serializeUnit(unit));
            }
        }
    } else {
        // For ENEMY tribes, only include units in observability grid
        for (Integer cityId : tribe.getCitiesID()) {
            City city = (City) board.getActor(cityId);
            if (city == null) {
                continue;
            }
            for (Integer unitId : city.getUnitsID()) {
                if (unitId == null || seen.contains(unitId)) {
                    continue;
                }
                seen.add(unitId);
                Unit unit = (Unit) board.getActor(unitId);
                if (unit != null) {
                    // CHECK: Is unit visible in observability grid?
                    int unitX = unit.getPosition().x;
                    int unitY = unit.getPosition().y;
                    if (isPositionVisible(obsGrid, unitX, unitY)) {
                        units.put(serializeUnit(unit));
                    }
                }
            }
        }
        // Include extra units (only if visible)
        for (Integer unitId : tribe.getExtraUnits()) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit != null) {
                int unitX = unit.getPosition().x;
                int unitY = unit.getPosition().y;
                if (isPositionVisible(obsGrid, unitX, unitY)) {
                    units.put(serializeUnit(unit));
                }
            }
        }
    }

    return units;
}
```

**Key changes:**
- Check if tribe is active
- If active: include all units (current behavior)
- If enemy: only include if visible (FOW check)
- New helper method: `isPositionVisible()`

---

### Step 4: Add visibility helper method

**Add this new method near other helper methods:**

```java
private static boolean isPositionVisible(boolean[][] obsGrid, int x, int y) {
    /**
     * Check if a position is visible in the observability grid.
     * Returns true if:
     * 1. Position is within grid bounds
     * 2. Position is marked as observable (obsGrid[x][y] == true)
     */
    if (obsGrid == null) {
        // No FOW restriction (obsGrid not set)
        return true;
    }
    if (x < 0 || y < 0 || x >= obsGrid.length) {
        return false;
    }
    if (y >= obsGrid[0].length) {
        return false;
    }
    return obsGrid[x][y];
}
```

---

### Step 5: Update `serializeCities()` method

**Location:** Line 175 in `src/players/PythonBridge.java`

**Current signature:**
```java
private static JSONArray serializeCities(Board board, Tribe tribe) {
```

**New signature:**
```java
private static JSONArray serializeCities(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
```

**Inside method, add FOW check:**

Current:
```java
private static JSONArray serializeCities(Board board, Tribe tribe) {
    JSONArray cities = new JSONArray();
    for (Integer cityId : tribe.getCitiesID()) {
        City city = (City) board.getActor(cityId);
        if (city == null) {
            continue;
        }
        // ... serialize city
        cities.put(c);
    }
    return cities;
}
```

Updated:
```java
private static JSONArray serializeCities(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
    JSONArray cities = new JSONArray();
    for (Integer cityId : tribe.getCitiesID()) {
        City city = (City) board.getActor(cityId);
        if (city == null) {
            continue;
        }
        
        // FOW CHECK: Only include city if visible (or if it's our tribe)
        if (tribe.getTribeId() != activeTribeID) {
            // Enemy city - check visibility
            int cityX = city.getPosition().x;
            int cityY = city.getPosition().y;
            if (!isPositionVisible(obsGrid, cityX, cityY)) {
                continue;  // Skip this hidden city
            }
        }
        
        JSONObject c = new JSONObject();
        // ... rest of serialization code
        cities.put(c);
    }
    return cities;
}
```

**Key change:**
- Before serializing enemy cities, check if they're visible
- Skip if not visible

---

## Testing the Fix

### Test 1: Verify compilation
```bash
javac -cp "src:lib/*" $(find src -name "*.java")
```
Should succeed without errors.

### Test 2: Run a quick game
```bash
java -cp "src:lib/*" Run  # or your runner
```

### Test 3: Inspect captures
```bash
# Look at the first capture
cat py_api/captures/capture_tick*.json | head -200

# Check for enemy units - should NOT be present if outside FOW range
```

Key things to check in capture:
- Active tribe (tribe_id = 0, say): ALL units should be present
- Enemy tribe (tribe_id = 1): Only units at positions that are in FOW
- Out-of-sight enemy units: Should NOT appear at all

### Test 4: Train and run
```bash
cd py_api
python train.py --max-samples 1  # Quick test
```

Should not have any import/encoding errors related to hidden unit positions.

---

## Verification Checklist

- [ ] `buildPayload()` extracts and passes `obsGrid` and `activeTribeID`
- [ ] `serializeTribes()` new signature accepts these parameters
- [ ] `serializeTribes()` passes parameters to helper methods
- [ ] `serializeTribeUnits()` checks `tribe.getTribeId() == activeTribeID`
- [ ] Enemy units are filtered by `isPositionVisible()`
- [ ] `serializeCities()` new signature accepts FOW parameters
- [ ] Enemy cities are filtered by visibility
- [ ] Helper method `isPositionVisible()` handles null/bounds cases
- [ ] Code compiles without errors
- [ ] Game runs without errors
- [ ] Captures don't contain out-of-sight enemy units
- [ ] Training script runs on new captures

---

## Common Pitfalls to Avoid

❌ **DON'T:** Filter active tribe's own units (active tribe has full visibility)
✅ **DO:** Only filter enemy tribe units

❌ **DON'T:** Remove all enemy data (still want to see tech, score, etc.)
✅ **DO:** Only filter position-based data (units, cities)

❌ **DON'T:** Forget to update method signatures in `serializeTribes()`
✅ **DO:** Pass `obsGrid` and `activeTribeID` to both helper methods

❌ **DON'T:** Add FOW filter to `serializeBoard()` (board terrain is generally known)
✅ **DO:** Only filter units/cities that appear on board

---

## Rollback Plan

If anything goes wrong:
1. The changes are all contained in PythonBridge.java
2. Undo changes and rebuild
3. Revert to earlier commit if needed

No other files are affected.

---

## Success Criteria

✅ Game runs without errors
✅ New captures don't include out-of-sight enemies
✅ Model can train on new captures
✅ No type/null pointer errors related to positions
✅ Game performance unchanged (no slowdown)

---

## Questions?

If stuck on specific implementation:
1. Check actual positions with `unit.getPosition()` and `city.getPosition()`
2. Verify `obsGrid` is not null before using
3. Ensure both X and Y are checked (2D grid)
4. Remember: obsGrid[x][y] might be false for unseen tiles

