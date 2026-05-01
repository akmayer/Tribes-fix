package players;

import core.actions.Action;
import core.game.GameState;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

public class RandomAgent extends Agent {

    private Random rnd;

    public RandomAgent(long seed)
    {
        super(seed);
        rnd = new Random(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect)
    {
        ArrayList<Action> allActions = gs.getAllAvailableActions();
        Action toExecute = null;
        try {
            String resp = PythonBridge.queryPolicy(gs, allActions);
            System.out.println("[PythonBridge response] " + resp);
        } catch (Exception e) {
            // If the bridge is not available, fall back to random action
            System.out.println("[PythonBridge] error: " + e.getMessage() + ". Falling back to random.");
        }

        int nActions = allActions.size();
        if (nActions > 0) {
            toExecute = allActions.get(rnd.nextInt(nActions));
        }
        return toExecute;
    }

    @Override
    public Agent copy() {
        return null;
    }
}
