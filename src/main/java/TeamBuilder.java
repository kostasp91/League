import java.io.IOException;
import java.util.*;

public class TeamBuilder {

    private static final double MAX_COST = 94.2;
    private static final int GUARD = 4;
    private static final int FORWARD = 4;
    private static final int CENTER = 2;
    private static final int TOP_PLAYERS = 30; // Only consider top 30 per position

    private static double bestScoreGlobal = -1;

    // Build best team
    public static List<Player> buildBestTeam(
            List<Player> players,
            Map<String, Integer> positionLimits,
            double maxCost,
            List<Player> fixedPlayers) {

        List<Player> bestTeam = new ArrayList<>();

        // --- Group players by position ---
        Map<String, List<Player>> posMap = new HashMap<>();
        for (Player p : players) {
            posMap.computeIfAbsent(p.getPos(), k -> new ArrayList<>()).add(p);
        }

        // --- Sort by score descending and prune to top N ---
        for (List<Player> list : posMap.values()) {
            list.sort(Comparator.comparingDouble(Player::getScore).reversed());
            if (list.size() > TOP_PLAYERS) {
                list.subList(TOP_PLAYERS, list.size()).clear();
            }
        }

        // --- Prepare current team and position usage ---
        List<Player> currentTeam = new ArrayList<>(fixedPlayers);
        Map<String, Integer> posUsed = new HashMap<>();
        double currentCost = 0.0;
        double currentScore = 0.0;

        for (Player p : fixedPlayers) {
            posUsed.put(p.getPos(), posUsed.getOrDefault(p.getPos(), 0) + 1);
            currentCost += p.getCost();
            currentScore += p.getScore();
        }

        bestScoreGlobal = -1;
        backtrack(posMap, positionLimits, currentScore, currentTeam, posUsed, currentCost, maxCost, bestTeam);

        return bestTeam;
    }

    private static void backtrack(Map<String, List<Player>> posMap,
                                  Map<String, Integer> positionLimits,
                                  double currentScore,
                                  List<Player> currentTeam,
                                  Map<String, Integer> posUsed,
                                  double currentCost,
                                  double maxCost,
                                  List<Player> bestTeam) {

        if (currentCost > maxCost) return;

        // Check if team is complete
        boolean complete = true;
        for (String pos : positionLimits.keySet()) {
            int used = posUsed.getOrDefault(pos, 0);
            if (used < positionLimits.get(pos)) {
                complete = false;
                break;
            }
        }

        if (complete && currentScore > bestScoreGlobal) {
            bestScoreGlobal = currentScore;
            bestTeam.clear();
            bestTeam.addAll(new ArrayList<>(currentTeam));
        }

        // Try adding next players
        for (String pos : posMap.keySet()) {
            int used = posUsed.getOrDefault(pos, 0);
            int limit = positionLimits.get(pos);
            if (used >= limit) continue;

            for (Player p : posMap.get(pos)) {
                if (currentTeam.contains(p)) continue;

                currentTeam.add(p);
                posUsed.put(pos, used + 1);

                backtrack(posMap, positionLimits,
                        currentScore + p.getScore(),
                        currentTeam,
                        posUsed,
                        currentCost + p.getCost(),
                        maxCost,
                        bestTeam);

                currentTeam.remove(currentTeam.size() - 1);
                posUsed.put(pos, used);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        List<Player> players = PlayerRecommender.getAllPlayers();
        Scanner scanner = new Scanner(System.in);

        System.out.println("Select mode: 1 = Print all players, 2 = Build optimal team");
        int mode = scanner.nextInt();

        if (mode == 1) {
            System.out.println("\nTop recommended players:");
            players.sort(Comparator.comparingDouble(Player::getScore).reversed());
            for (int i = 0; i < Math.min(TOP_PLAYERS, players.size()); i++) {
                System.out.println(players.get(i));
            }
        } else if (mode == 2) {
            Map<String, Integer> positionLimits = new HashMap<>();
            positionLimits.put("G", GUARD);
            positionLimits.put("F", FORWARD);
            positionLimits.put("C", CENTER);

            // Must-have players
            List<String> mustHaveNames = Arrays.asList(
                    "S. Vezenkov",
                    "N. Milutinov",
                    "J. Nwora",
                    "S. De Larrea"
                    //, "E. Osmani"
            );

            List<Player> fixedPlayers = new ArrayList<>();
            for (String name : mustHaveNames) {
                players.stream()
                        .filter(p -> p.getName().equalsIgnoreCase(name))
                        .findFirst()
                        .ifPresentOrElse(
                                fixedPlayers::add,
                                () -> System.out.println("Player not found: " + name)
                        );
            }

            double lockedCost = fixedPlayers.stream().mapToDouble(Player::getCost).sum();
            System.out.println("\nLocked Players:");
            for (Player p : fixedPlayers) {
                System.out.printf(" - %s (%s) | Cost: %.2f | Score: %.2f%n", p.getName(), p.getPos(), p.getCost(), p.getScore());
            }
            System.out.printf("Total locked cost: %.2f / %.2f%n", lockedCost, MAX_COST);

            // Build best team
            List<Player> bestTeam = buildBestTeam(players, positionLimits, MAX_COST, fixedPlayers);

            // Print optimal team
            System.out.println("\nOptimal Team:");
            double totalCost = 0;
            double totalScore = 0;
            for (Player p : bestTeam) {
                totalCost += p.getCost();
                totalScore += p.getScore();
                System.out.println(p);
            }
            System.out.printf("Total cost: %.2f / %.2f | Total score: %.2f%n", totalCost, MAX_COST, totalScore);
        } else {
            System.out.println("Invalid mode.");
        }
    }
}
