import core.Types;
import core.game.Game;
import org.json.JSONArray;
import players.*;
import gui.GUI;
import gui.WindowInput;
import players.emcts.EMCTSAgent;
import players.emcts.EMCTSParams;
import players.heuristics.PrunePortfolioHeuristic;
import players.mc.MCParams;
import players.mc.MonteCarloAgent;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;
import players.oep.OEPAgent;
import players.oep.OEPParams;
import players.osla.OSLAParams;
import players.osla.OneStepLookAheadAgent;
import players.portfolio.Portfolio;
import players.portfolio.SimplePortfolio;
import players.portfolioMCTS.PortfolioMCTSParams;
import players.portfolioMCTS.PortfolioMCTSPlayer;
import players.rhea.RHEAAgent;
import players.rhea.RHEAParams;
import java.awt.GraphicsEnvironment;

import static core.Constants.*;
import static core.Types.TRIBE.*;

class Run {

    /**
     * Runs 1 game.
     * @param g - game to run
     * @param ki - Key controller
     * @param ac - Action controller
     */
    static void runGame(Game g, KeyController ki, ActionController ac) {
        WindowInput wi = null;
        GUI frame = null;
        if (VISUALS && !GraphicsEnvironment.isHeadless()) {
            wi = new WindowInput();
            wi.windowClosed = false;
            frame = new GUI(g, "Tribes", ki, wi, ac, false);
            frame.addWindowListener(wi);
            frame.addKeyListener(ki);
        } else if (VISUALS) {
            System.out.println("Running without visuals because no display is available.");
        }

        g.run(frame, wi);
    }


    /**
     * Runs a game, no visuals nor human player
     * @param g - game to run
     */
    static void runGame(Game g) {
        g.run(null, null);
    }


    public enum PlayerType
    {
        DONOTHING,
        HUMAN,
        RANDOM,
        OSLA,
        MC,
        SIMPLE,
        POLICY,
        MCTS,
        AZ_MCTS,
        RHEA,
        OEP,
        EMCTS,
        PORTFOLIO_MCTS
    }

    public static double K_INIT_MULT = 0.5;
    public static double T_MULT = 2.0;
    public static double A_MULT = 1.5;
    public static double B = 1.3;
    public static double[] pMCTSweights;

    public static int MAX_LENGTH;
    public static boolean PRUNING;
    public static boolean PROGBIAS;
    public static boolean FORCE_TURN_END;
    public static boolean MCTS_ROLLOUTS;
    public static int POP_SIZE;


    static Run.PlayerType parsePlayerTypeStr(String arg) throws Exception
    {
        switch(arg)
        {
            case "Human": return Run.PlayerType.HUMAN;
            case "Do Nothing": return Run.PlayerType.DONOTHING;
            case "Random": return Run.PlayerType.RANDOM;
            case "Rule Based": return Run.PlayerType.SIMPLE;
            case "Policy":
            case "POLICY": return Run.PlayerType.POLICY;
            case "OSLA": return Run.PlayerType.OSLA;
            case "MC": return Run.PlayerType.MC;
            case "MCTS": return Run.PlayerType.MCTS;
            case "AZ_MCTS": return Run.PlayerType.AZ_MCTS;
            case "RHEA": return Run.PlayerType.RHEA;
            case "OEP": return Run.PlayerType.OEP;
            case "pMCTS": return Run.PlayerType.PORTFOLIO_MCTS;
            case "EMCTS": return Run.PlayerType.EMCTS;
        }
        throw new Exception("Error: unrecognized Player Type: " + arg);
    }

    static Types.TRIBE parseTribeStr(String arg) throws Exception
    {
        switch(arg)
        {
            case "Xin Xi": return XIN_XI;
            case "Imperius": return IMPERIUS;
            case "Bardur": return BARDUR;
            case "Oumaji": return OUMAJI;
            case "Kickoo": return KICKOO;
            case "Hoodrick": return HOODRICK;
            case "Luxidoor": return LUXIDOOR;
            case "Vengir": return VENGIR;
            case "Zebasi": return ZEBASI;
            case "Ai-Mo": return AI_MO;
            case "Quetzali": return QUETZALI;
            case "Yadakk": return YADAKK;
        }
        throw new Exception("Error: unrecognized Tribe: " + arg);
    }

    public static double[] getWeights(JSONArray w) {
        if (w == null) return null;
        double[] weights = new double[w.length()];
        for (int i = 0; i < weights.length; ++i)
        {
            weights[i] = w.getDouble(i);
        }
        return weights;
    }

    private static int envInt(String name, int defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            System.out.println("Invalid integer env " + name + "=" + value + "; using " + defaultValue);
            return defaultValue;
        }
    }

    private static double envDouble(String name, double defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            System.out.println("Invalid double env " + name + "=" + value + "; using " + defaultValue);
            return defaultValue;
        }
    }

    private static boolean envBoolean(String name, boolean defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        return value.equalsIgnoreCase("true") || value.equals("1") || value.equalsIgnoreCase("yes");
    }

    public static Agent getAgent(Run.PlayerType playerType, long agentSeed)
    {
        switch (playerType)
        {
            case DONOTHING: return new DoNothingAgent(agentSeed);
            case RANDOM: return new BridgeAgentAttempt(agentSeed);
            case SIMPLE: return new SimpleAgent(agentSeed);
            case POLICY: return new NeuralPolicyAgent(agentSeed);
            case OSLA:
                OSLAParams oslaParams = new OSLAParams();
                oslaParams.stop_type = oslaParams.STOP_FMCALLS; //Upper bound
                oslaParams.heuristic_method = oslaParams.DIFF_HEURISTIC;
                return new OneStepLookAheadAgent(agentSeed, oslaParams);
            case MC:
                MCParams mcparams = new MCParams();
                mcparams.stop_type = mcparams.STOP_FMCALLS;
                mcparams.heuristic_method = mcparams.DIFF_HEURISTIC;
                mcparams.PRIORITIZE_ROOT = true;
                mcparams.ROLLOUT_LENGTH = MAX_LENGTH;
                mcparams.FORCE_TURN_END = FORCE_TURN_END ? 5 : mcparams.ROLLOUT_LENGTH + 1;
                return new MonteCarloAgent(agentSeed, mcparams);
            case MCTS:
                MCTSParams mctsParams = new MCTSParams();
                mctsParams.stop_type = mctsParams.STOP_FMCALLS;
                mctsParams.heuristic_method = mctsParams.DIFF_HEURISTIC;
                mctsParams.PRIORITIZE_ROOT = true;
                mctsParams.ROLLOUT_LENGTH = MAX_LENGTH;
                mctsParams.FORCE_TURN_END = FORCE_TURN_END ? 5 : mctsParams.ROLLOUT_LENGTH + 1;
                mctsParams.ROLOUTS_ENABLED = MCTS_ROLLOUTS;
                return new MCTSPlayer(agentSeed, mctsParams);
            case AZ_MCTS:
                MCTSParams azParams = new MCTSParams();
                azParams.stop_type = azParams.STOP_ITERATIONS;
                azParams.heuristic_method = azParams.DIFF_HEURISTIC;
                azParams.PRIORITIZE_ROOT = false;
                azParams.ROLLOUT_LENGTH = MAX_LENGTH;
                azParams.FORCE_TURN_END = FORCE_TURN_END ? 5 : azParams.ROLLOUT_LENGTH + 1;
                azParams.ROLOUTS_ENABLED = false;
                azParams.CAPTURE_MCTS = envBoolean("TRIBES_AZ_CAPTURE_MCTS", true);
                azParams.NEURAL_PRIORS = true;
                azParams.NEURAL_VALUE = true;
                azParams.num_iterations = envInt("TRIBES_AZ_MCTS_SIMULATIONS", 128);
                azParams.CPUCT = envDouble("TRIBES_AZ_MCTS_CPUCT", 1.5);
                azParams.DIRICHLET_ROOT_NOISE = envBoolean("TRIBES_AZ_DIRICHLET_ROOT_NOISE", true);
                azParams.DIRICHLET_ALPHA = envDouble("TRIBES_AZ_DIRICHLET_ALPHA", 0.30);
                azParams.DIRICHLET_EPSILON = envDouble("TRIBES_AZ_DIRICHLET_EPSILON", 0.25);
                azParams.FORCE_END_TURN_IN_SEARCH = envBoolean("TRIBES_AZ_FORCE_END_TURN_IN_SEARCH", false);
                azParams.DEBUG_DECISIONS = envBoolean("TRIBES_AZ_DEBUG_DECISIONS", false);
                azParams.MASK_SEND_STARS = envBoolean("TRIBES_MASK_SEND_STARS", true);
                return new MCTSPlayer(agentSeed, azParams);
            case PORTFOLIO_MCTS:
                PortfolioMCTSParams portfolioMCTSParams = new PortfolioMCTSParams();
                portfolioMCTSParams.stop_type = portfolioMCTSParams.STOP_FMCALLS;
                portfolioMCTSParams.heuristic_method = portfolioMCTSParams.DIFF_HEURISTIC;
                portfolioMCTSParams.PRIORITIZE_ROOT = false;
                portfolioMCTSParams.ROLLOUT_LENGTH = MAX_LENGTH;
                portfolioMCTSParams.PRUNING = PRUNING;
                portfolioMCTSParams.PROGBIAS = PROGBIAS;
                portfolioMCTSParams.K_init_mult = K_INIT_MULT;
                portfolioMCTSParams.A_mult = A_MULT;
                portfolioMCTSParams.B = B;
                portfolioMCTSParams.T_mult = T_MULT;
                Portfolio p = new SimplePortfolio(agentSeed);
                portfolioMCTSParams.setPortfolio(p);
                portfolioMCTSParams.pruneHeuristic = new PrunePortfolioHeuristic(p);
                if(Run.pMCTSweights != null)
                    portfolioMCTSParams.pruneHeuristic.setWeights(Run.pMCTSweights);
                return new PortfolioMCTSPlayer(agentSeed, portfolioMCTSParams);
            case OEP:
                OEPParams oepParams = new OEPParams();
                oepParams.stop_type = oepParams.STOP_FMCALLS;
                oepParams.heuristic_method = oepParams.DIFF_HEURISTIC;
                return new OEPAgent(agentSeed, oepParams);
            case EMCTS:
                EMCTSParams emctsParams = new EMCTSParams();
                emctsParams.stop_type = emctsParams.STOP_FMCALLS;
                emctsParams.heuristic_method = emctsParams.DIFF_HEURISTIC;
                return new EMCTSAgent(agentSeed,emctsParams);
            case RHEA:
                RHEAParams rheaParams = new RHEAParams();
                rheaParams.stop_type = rheaParams.STOP_FMCALLS;
                rheaParams.heuristic_method = rheaParams.DIFF_HEURISTIC;
                rheaParams.INDIVIDUAL_LENGTH = MAX_LENGTH;
                rheaParams.FORCE_TURN_END = rheaParams.INDIVIDUAL_LENGTH + 1;
                rheaParams.POP_SIZE = POP_SIZE;
                return new RHEAAgent(agentSeed, rheaParams);
        }
        return null;
    }
}
