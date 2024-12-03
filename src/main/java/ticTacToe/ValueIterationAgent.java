package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

    /**
     * This map is used to store the values of states
     */
    Map<Game, Double> valueFunction = new HashMap<Game, Double>();
    
    /**
     * the discount factor
     */
    double discount = 0.9;
    
    /**
     * the MDP model
     */
    TTTMDP mdp = new TTTMDP();
    
    /**
     * the number of iterations to perform - feel free to change this/try out different numbers of iterations
     */
    int k = 10;
    
    
    /**
     * This constructor trains the agent offline first and sets its policy
     */
    public ValueIterationAgent()
    {
        super();
        mdp = new TTTMDP();
        this.discount = 0.9;
        initValues();
        train();
    }
    
    
    /**
     * Use this constructor to initialise your agent with an existing policy
     * @param p
     */
    public ValueIterationAgent(Policy p) {
        super(p);
        
    }

    public ValueIterationAgent(double discountFactor) {
        
        this.discount = discountFactor;
        mdp = new TTTMDP();
        initValues();
        train();
    }
    
    /**
     * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
     * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
     * 
     */
    public void initValues()
    {
        
        List<Game> allGames = Game.generateAllValidGames('X'); // all valid games where it is X's turn, or it's terminal.
        for(Game g: allGames)
            this.valueFunction.put(g, 0.0);
        
        
        
    }
    
    
    
    public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
    {
        this.discount = discountFactor;
        mdp = new TTTMDP(winReward, loseReward, livingReward, drawReward);
    }
    
    /*
     * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
     * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
     */
    public void iterate() {
        // Loop through the specified number of steps (iterations)
        for (int step = 0; step < k; step++) {
            // Create a temporary map to store updated state values
            Map<Game, Double> updatedValues = new HashMap<>();

            // Iterate through each game state in the current value function
            for (Game state : valueFunction.keySet()) {
                // If the state is terminal, its value is always 0
                if (state.isTerminal()) {
                    updatedValues.put(state, 0.0);
                    continue;
                }

                // Initialize the maximum Q-value for this state to a very negative value
                double maxQValue = -Double.MAX_VALUE;

                // Iterate through all possible moves for the current state
                for (Move action : state.getPossibleMoves()) {
                    double qValue = 0.0;

                    // Calculate the Q-value for the move by summing over all possible transitions
                    for (TransitionProb transition : mdp.generateTransitions(state, action)) {
                        qValue += transition.prob * (transition.outcome.localReward + discount * valueFunction.get(transition.outcome.sPrime));
                    }

                    // Update the maximum Q-value for this state
                    maxQValue = Math.max(maxQValue, qValue);
                }

                // Store the updated maximum Q-value for the state
                updatedValues.put(state, maxQValue);
            }

            // Apply the updated values to the value function for the next iteration
            valueFunction.putAll(updatedValues);
        }
    }

    
    /**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
     * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
     * to extract a policy.
     * 
     * @return the policy according to {@link ValueIterationAgent#valueFunction}
     */
    public Policy extractPolicy() {
        // Create a new policy object to store the best moves for each state
        Policy policy = new Policy();

        // Iterate through each state in the value function
        for (Game state : valueFunction.keySet()) {
            // Skip terminal states as they don't require any moves
            if (state.isTerminal()) {
                continue;
            }

            // Initialize variables to track the best move and its corresponding Q-value
            Move bestMove = null;
            double maxQValue = -Double.MAX_VALUE;

            // Iterate through all possible moves for the current state
            for (Move action : state.getPossibleMoves()) {
                double qValue = 0.0;

                // Calculate the Q-value for the current move by summing over all possible transitions
                for (TransitionProb transition : mdp.generateTransitions(state, action)) {
                    qValue += transition.prob * (transition.outcome.localReward + discount * valueFunction.get(transition.outcome.sPrime));
                }

                // Update the best move if the current Q-value is higher than the previous maximum
                if (qValue > maxQValue) {
                    maxQValue = qValue;
                    bestMove = action;
                }
            }

            // Add the best move for the current state to the policy
            policy.policy.put(state, bestMove);
        }

        // Return the extracted policy containing the best moves for all states
        return policy;
    }

    /**
     * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
     * {@link ValueIterationAgent#iterate}. 
     */
    public void train()
    {
        /**
         * First run value iteration
         */
        this.iterate();
        /**
         * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
         *  
         */
        
        super.policy = extractPolicy();
        
        if (this.policy == null)
        {
            System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
            //System.exit(1);
        }
    }

    public static void main(String a[]) throws IllegalMoveException
    {
        //Test method to play the agent against a human agent.
        ValueIterationAgent agent = new ValueIterationAgent();
        HumanAgent d = new HumanAgent();
        
        Game g = new Game(agent, d, d);
        g.playOut();
    }
}
