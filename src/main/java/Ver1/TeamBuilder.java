package Ver1;

import java.io.*;
import java.util.*;

public class TeamBuilder {

    private static final double MAX_COST = 95;
    private static final int GUARD = 4;
    private static final int FORWARD = 4;
    private static final int CENTER = 2;
    private static final int TOP_PLAYERS = 30;
    private static final String OUTPUTFILENAME = "optimal_team.txt";

    private static double bestScoreGlobal = -1;

    //
    // Build best team with optional fixed players
    //
    public static List<Player> buildBestTeam(
            List<Player> players,
            Map<String, Integer> positionLimits,
            double maxCost,
            List<Player> fixedPlayers) {

        List<Player> bestTeam = new ArrayList<>();

        // Group players by position
        Map<String, List<Player>> posMap = new HashMap<>();
        for (Player p : players) {
            posMap.computeIfAbsent(p.getPos(), k -> new ArrayList<>()).add(p);
        }

        // Sort by score descending and prune to top N
        for (List<Player> list : posMap.values()) {
            list.sort(Comparator.comparingDouble(Player::getScore).reversed());
            if (list.size() > TOP_PLAYERS) {
                list.subList(TOP_PLAYERS, list.size()).clear();
            }
        }

        // Prepare current team and position usage
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

    //
    // Recursive team builder
    //
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

    //
    // Save results to file
    //
    private static void saveTeamToFile(List<Player> team, double totalCost, double totalScore) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(OUTPUTFILENAME))) {
            writer.println("Optimal Team:");
            for (Player p : team) {
                writer.println(p); // Uses Player.toString()
            }
            writer.printf("%nTotal cost: %.2f / %.2f | Total score: %.2f%n",
                    totalCost, MAX_COST, totalScore);
            System.out.println("\nTeam also saved to '" + OUTPUTFILENAME + "'");
        } catch (IOException e) {
            System.err.println("Error saving team to file: " + e.getMessage());
        }
    }

    //
    // Main Method
    //
    public static void main(String[] args) throws IOException {
        List<Player> players = PlayerRecommender.getAllPlayers();
        Scanner scanner = new Scanner(System.in);

        System.out.println("Select mode: 1 = Print all players, 2 = Build optimal team");
        int mode = scanner.nextInt();

        if (mode == 1) {
            System.out.println("Select position: G = Guard, F = Forward, C = Center, A = All");
            String position = scanner.next().toUpperCase();

            System.out.println("\nTop " + position + " recommended players:");
            players.sort(Comparator.comparingDouble(Player::getScore).reversed());

            List<Player> filteredPlayers;
            if (position.equals("A")) {
                filteredPlayers = players;
            } else {
                filteredPlayers = new ArrayList<>();
                for (Player p : players) {
                    if (p.getPos().equalsIgnoreCase(position)) {
                        filteredPlayers.add(p);
                    }
                }
            }

            filteredPlayers.sort(Comparator.comparingDouble(Player::getScore).reversed());
            for (int i = 0; i < Math.min(TOP_PLAYERS, filteredPlayers.size()); i++) {
                System.out.printf("%2d. %s%n", i + 1, filteredPlayers.get(i));
            }

            if (filteredPlayers.isEmpty()) {
                System.out.println("No players found for position: " + position);
            }

        } else if (mode == 2) {
            Map<String, Integer> positionLimits = new HashMap<>();
            positionLimits.put("G", GUARD);
            positionLimits.put("F", FORWARD);
            positionLimits.put("C", CENTER);

            // Must-have players (Name|Position)
            List<String> mustHaveEntries = Arrays.asList(
                    "S. Vezenkov|F",
                    "N. Milutinov|C",
                    "J. Nwora|F",
                    "E. Osmani|F",
                    "S. De Larrea|G",
                    "D. Hall|G",
                    "J. Loyd|G"

            );

            List<Player> fixedPlayers = new ArrayList<>();
            for (String entry : mustHaveEntries) {
                String[] parts = entry.split("\\|");
                if (parts.length != 2) {
                    System.out.println("Invalid must-have format: " + entry);
                    continue;
                }

                String name = parts[0].trim();
                String pos = parts[1].trim().toUpperCase();

                players.stream()
                        .filter(p -> p.getName().equalsIgnoreCase(name)
                                && p.getPos().equalsIgnoreCase(pos))
                        .findFirst()
                        .ifPresentOrElse(
                                fixedPlayers::add,
                                () -> System.out.println("Player not found: " + name + " (" + pos + ")")
                        );
            }

            double lockedCost = fixedPlayers.stream().mapToDouble(Player::getCost).sum();
            System.out.println("\nLocked Players:");
            for (Player p : fixedPlayers) {
                System.out.println(p);
            }
            System.out.printf("Total locked cost: %.2f / %.2f%n", lockedCost, MAX_COST);

            // Build best team
            List<Player> bestTeam = buildBestTeam(players, positionLimits, MAX_COST, fixedPlayers);

            // Print optimal team
            System.out.println("\nOptimal Team:");
            double totalCost = 0;
            double totalScore = 0;
            double sumAvgFantasyPoints = 0;
            for (Player p : bestTeam) {
                totalCost += p.getCost();
                totalScore += p.getScore();
                sumAvgFantasyPoints += p.getAvgFantasyPoints();
                System.out.println(p);
            }

            System.out.printf("\nTotal cost: %.2f / %.2f | Sum AVG Fantasy Points %.2f | Total score: %.2f%n",
                    totalCost, MAX_COST, sumAvgFantasyPoints, totalScore);

            saveTeamToFile(bestTeam, totalCost, totalScore);
        } else {
            System.out.println("Invalid mode.");
        }
    }
}
