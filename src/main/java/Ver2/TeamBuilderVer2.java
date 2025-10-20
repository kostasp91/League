package Ver2;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class TeamBuilderVer2 {

    private static final double MAX_COST = 89.5;
    private static final int GUARD = 4;
    private static final int FORWARD = 4;
    private static final int CENTER = 2;
    private static final int TOP_PLAYERS = 30;
    private static final String OUTPUTFILENAME = "optimal_team_ver2.txt";
    private static final String OUTPUTFILENAME_WITH_SUBS = "optimal_team_ver2_with_subs.txt";

    private static double bestScoreGlobal = -1;

    // ====== Core builder (same behavior as earlier today) ======
    public static List<PlayerVer2> buildBestTeam(
            List<PlayerVer2> players,
            Map<String, Integer> positionLimits,
            double maxCost,
            List<PlayerVer2> fixedPlayers) {

        List<PlayerVer2> bestTeam = new ArrayList<>();

        Map<String, List<PlayerVer2>> posMap = new HashMap<>();
        for (PlayerVer2 p : players) {
            posMap.computeIfAbsent(p.getPos().toUpperCase(), k -> new ArrayList<>()).add(p);
        }

        // Order by efficiency (original behavior); trim search space
        for (List<PlayerVer2> list : posMap.values()) {
            list.sort(Comparator.comparingDouble(PlayerVer2::getEfficiency).reversed());
            if (list.size() > TOP_PLAYERS) list.subList(TOP_PLAYERS, list.size()).clear();
        }

        List<PlayerVer2> currentTeam = new ArrayList<>(fixedPlayers);
        Map<String, Integer> posUsed = new HashMap<>();
        double currentCost = 0.0;
        double currentPoints = 0.0;

        for (PlayerVer2 p : fixedPlayers) {
            String pos = p.getPos().toUpperCase();
            posUsed.put(pos, posUsed.getOrDefault(pos, 0) + 1);
            currentCost += p.getCost();
            currentPoints += p.predictedNextFantasyPoints;
        }

        bestScoreGlobal = -1;
        backtrack(posMap, positionLimits, currentPoints, currentTeam, posUsed, currentCost, maxCost, bestTeam);

        return bestTeam;
    }

    private static void backtrack(Map<String, List<PlayerVer2>> posMap,
                                  Map<String, Integer> positionLimits,
                                  double currentPoints,
                                  List<PlayerVer2> currentTeam,
                                  Map<String, Integer> posUsed,
                                  double currentCost,
                                  double maxCost,
                                  List<PlayerVer2> bestTeam) {

        if (currentCost > maxCost) return;

        boolean complete = true;
        for (String pos : positionLimits.keySet()) {
            int used = posUsed.getOrDefault(pos, 0);
            if (used < positionLimits.get(pos)) { complete = false; break; }
        }

        if (complete && currentPoints > bestScoreGlobal) {
            bestScoreGlobal = currentPoints;
            bestTeam.clear();
            bestTeam.addAll(new ArrayList<>(currentTeam));
            return;
        }

        // Try to fill any position that still needs players
        for (String pos : posMap.keySet()) {
            int used = posUsed.getOrDefault(pos, 0);
            int limit = positionLimits.get(pos);
            if (used >= limit) continue;

            // Iterate over a snapshot to be safe from ConcurrentModification
            List<PlayerVer2> candidates = posMap.get(pos);
            for (int i = 0; i < candidates.size(); i++) {
                PlayerVer2 p = candidates.get(i);
                if (currentTeam.contains(p)) continue; // prevent duplicates

                currentTeam.add(p);
                posUsed.put(pos, used + 1);

                backtrack(posMap, positionLimits,
                        currentPoints + p.predictedNextFantasyPoints,
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

    // ====== Mode 3: Upgrade team with up to K substitutions ======

    private static int rosterSize(Map<String, Integer> positionLimits) {
        int sum = 0;
        for (int v : positionLimits.values()) sum += v;
        return sum;
    }

    private static List<PlayerVer2> loadTeamFromFile(String filename, List<PlayerVer2> universe) {
        List<PlayerVer2> team = new ArrayList<>();
        Map<String, List<PlayerVer2>> indexByName = new HashMap<>();
        for (PlayerVer2 p : universe) {
            indexByName.computeIfAbsent(p.getName().toLowerCase(), k -> new ArrayList<>()).add(p);
        }

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (line.startsWith("Optimal Team")) continue;
                if (line.startsWith("Total cost:")) break;

                // expected lines like: "S. Vezenkov | F | Pred FP: ..."
                String[] parts = line.split("\\|");
                if (parts.length < 2) continue;
                String name = parts[0].trim();
                String pos  = parts[1].trim().toUpperCase();

                List<PlayerVer2> candidates = indexByName.getOrDefault(name.toLowerCase(), Collections.emptyList());
                PlayerVer2 match = null;
                for (PlayerVer2 p : candidates) {
                    if (p.getPos().equalsIgnoreCase(pos)) { match = p; break; }
                }
                if (match == null && !candidates.isEmpty()) {
                    // fallback: ignore position if we have only one
                    if (candidates.size() == 1) match = candidates.get(0);
                }
                if (match != null && !team.contains(match)) {
                    team.add(match);
                }
            }
        } catch (IOException e) {
            System.err.println("Failed to read " + filename + ": " + e.getMessage());
        }
        return team;
    }

    private static double totalCost(List<PlayerVer2> team) {
        return team.stream().mapToDouble(PlayerVer2::getCost).sum();
    }

    private static double totalPoints(List<PlayerVer2> team) {
        return team.stream().mapToDouble(p -> p.predictedNextFantasyPoints).sum();
    }

    private static List<int[]> combinationsIndices(int n, int k) {
        List<int[]> res = new ArrayList<>();
        if (k < 0 || k > n) return res;
        int[] comb = new int[k];
        // init first combination
        for (int i = 0; i < k; i++) comb[i] = i;
        while (true) {
            res.add(comb.clone());
            // next
            int i;
            for (i = k - 1; i >= 0; i--) {
                if (comb[i] != i + n - k) break;
            }
            if (i < 0) break;
            comb[i]++;
            for (int j = i + 1; j < k; j++) comb[j] = comb[j - 1] + 1;
        }
        return res;
    }

    private static void saveTeamToFile(List<PlayerVer2> team, double totalCost, double totalPoints, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("Optimal Team (availability-weighted efficiency):");
            for (PlayerVer2 p : team) writer.println(p);
            writer.printf("%nTotal cost: %.2f / %.2f | Total predicted FP: %.2f%n",
                    totalCost, MAX_COST, totalPoints);
            System.out.println("\n Team saved to '" + filename + "'");
        } catch (IOException e) {
            System.err.println("Error saving team to file: " + e.getMessage());
        }
    }

    private static void saveUpgradedTeamToFile(List<PlayerVer2> upgraded, List<PlayerVer2> base,
                                               String filename) {
        double tc = totalCost(upgraded);
        double tp = totalPoints(upgraded);
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("Upgraded Team (≤ 4 substitutions, maximize next week FP):");
            for (PlayerVer2 p : upgraded) writer.println(p);
            writer.printf("%nTotal cost: %.2f / %.2f | Total predicted FP: %.2f%n%n",
                    tc, MAX_COST, tp);

            // substitutions
            Set<PlayerVer2> upSet = new HashSet<>(upgraded);
            Set<PlayerVer2> baseSet = new HashSet<>(base);

            List<PlayerVer2> out = base.stream().filter(p -> !upSet.contains(p)).collect(Collectors.toList());
            List<PlayerVer2> in  = upgraded.stream().filter(p -> !baseSet.contains(p)).collect(Collectors.toList());

            writer.println("Substitutions:");
            writer.println("Out:");
            if (out.isEmpty()) writer.println("  - (none)");
            else for (PlayerVer2 p : out) writer.println("  - " + p);

            writer.println("In:");
            if (in.isEmpty()) writer.println("  - (none)");
            else for (PlayerVer2 p : in) writer.println("  - " + p);

            writer.println("Total subs: " + in.size());
            System.out.println("\n Team saved to '" + filename + "'");
        } catch (IOException e) {
            System.err.println("Error saving upgraded team to file: " + e.getMessage());
        }
    }

    private static boolean withinLimits(List<PlayerVer2> team, Map<String,Integer> limits) {
        Map<String,Integer> cnt = new HashMap<>();
        for (PlayerVer2 p : team) {
            String pos = p.getPos().toUpperCase();
            cnt.put(pos, cnt.getOrDefault(pos, 0) + 1);
        }
        for (String pos : limits.keySet()) {
            if (cnt.getOrDefault(pos, 0) != limits.get(pos)) return false;
        }
        return true;
    }

    // Build the best team keeping exactly keepCount players from baseTeam (=> substitutions = rosterSize - keepCount)
    private static List<PlayerVer2> tryWithKeepCount(List<PlayerVer2> universe,
                                                     Map<String,Integer> posLimits,
                                                     double budget,
                                                     List<PlayerVer2> baseTeam,
                                                     int keepCount) {
        int n = baseTeam.size();
        if (keepCount > n) keepCount = n;
        if (keepCount < 0) keepCount = 0;

        // choose which players to KEEP
        List<int[]> keepIdxSets = combinationsIndices(n, keepCount);

        List<PlayerVer2> best = new ArrayList<>();
        double bestPts = -1;

        for (int[] idxs : keepIdxSets) {
            List<PlayerVer2> keep = new ArrayList<>(keepCount);
            Set<PlayerVer2> keepSet = new HashSet<>();
            for (int i : idxs) {
                PlayerVer2 p = baseTeam.get(i);
                if (!keepSet.contains(p)) {
                    keep.add(p);
                    keepSet.add(p);
                }
            }

            // If keeping exceeds pos limits or budget, skip early
            if (!partialWithinLimits(keep, posLimits)) continue;
            double keepCost = totalCost(keep);
            if (keepCost > budget) continue;

            List<PlayerVer2> candidate = buildBestTeam(universe, posLimits, budget, keep);
            if (candidate.isEmpty()) continue;
            double pts = totalPoints(candidate);

            // sanity: require exactly roster size
            if (!withinLimits(candidate, posLimits)) continue;

            if (pts > bestPts) {
                bestPts = pts;
                best = candidate;
            }
        }
        return best;
    }

    private static boolean partialWithinLimits(List<PlayerVer2> partial, Map<String,Integer> limits) {
        Map<String,Integer> cnt = new HashMap<>();
        for (PlayerVer2 p : partial) {
            String pos = p.getPos().toUpperCase();
            cnt.put(pos, cnt.getOrDefault(pos, 0) + 1);
        }
        for (String pos : limits.keySet()) {
            if (cnt.getOrDefault(pos, 0) > limits.get(pos)) return false;
        }
        return true;
    }

    private static Double parseDoubleFlexible(String s) {
        if (s == null) return null;
        s = s.trim();
        if (s.isEmpty()) return null;
        s = s.replace(",", "."); // support European decimal comma
        try { return Double.parseDouble(s); }
        catch (NumberFormatException e) { return null; }
    }


    // ====== Main (adds Mode 3, keeps Modes 1 & 2 same as earlier) ======
    public static void main(String[] args) {
        try {
            System.out.println("Fetching player predictions from Flask model...");
            List<PlayerRecommenderVer2.Player> rawPlayers = PlayerRecommenderVer2.loadPlayers();
            PlayerRecommenderVer2.fetchPredictions(rawPlayers);

            List<PlayerVer2> players = new ArrayList<>();
            for (PlayerRecommenderVer2.Player p : rawPlayers) {
                players.add(new PlayerVer2(
                        p.name, p.team, p.pos,
                        p.avgFPT, p.predictedFPT, p.cost,
                        p.expectedMinutes, p.matchesPlayed, p.matchesTotal, p.pPlay
                ));
            }

            Scanner scanner = new Scanner(System.in);
            System.out.println("Select mode: 1 = Print top players, 2 = Build optimal team, 3 = Upgrade team (0–4 subs)");
            int mode = scanner.nextInt();
            scanner.nextLine();

            if (mode == 1) {
                System.out.println("Select position: G = Guard, F = Forward, C = Center, A = All");
                String position = scanner.nextLine().trim().toUpperCase(Locale.ROOT);

                List<PlayerVer2> filteredPlayers = new ArrayList<>();

                if (position.equals("A")) {
                    // All players, no cost filter
                    filteredPlayers = new ArrayList<>(players);

                    // Sort by predicted FP desc
                    filteredPlayers.sort(Comparator.comparingDouble(p -> -p.predictedNextFantasyPoints));

                    System.out.println("\nTop " + TOP_PLAYERS + " predicted players:");
                    for (int i = 0; i < Math.min(TOP_PLAYERS, filteredPlayers.size()); i++) {
                        System.out.println(filteredPlayers.get(i));
                    }
                } else {
                    // Filter by position first
                    for (PlayerVer2 pl : players) {
                        if (pl.getPos().equalsIgnoreCase(position)) filteredPlayers.add(pl);
                    }

                    if (filteredPlayers.isEmpty()) {
                        System.out.println("\nNo players found for position: " + position);
                        return;
                    }

                    // Ask for optional max cost cap
                    System.out.print("Max cost cap (e.g., 12.5). Press Enter for no cap: ");
                    String capStr = scanner.nextLine();
                    Double cap = parseDoubleFlexible(capStr);

                    if (cap != null) {
                        // Keep players with cost <= cap
                        final double capVal = cap;
                        filteredPlayers.removeIf(p -> p.getCost() > capVal + 1e-9);
                    }

                    if (filteredPlayers.isEmpty()) {
                        System.out.println("\nNo players match the given max cost cap.");
                        return;
                    }

                    // Sort by predicted FP desc
                    filteredPlayers.sort(Comparator.comparingDouble(p -> -p.predictedNextFantasyPoints));

                    // Print ALL players that match (not just top N)
                    System.out.println("\nPlayers (position " + position + (cap != null ? (", cost ≤ " + cap) : "") + "):");
                    for (PlayerVer2 p : filteredPlayers) {
                        System.out.println(p);
                    }
                    System.out.println("Total: " + filteredPlayers.size());
                }




        } else if (mode == 2) {
                Map<String, Integer> positionLimits = new HashMap<>();
                positionLimits.put("G", GUARD);
                positionLimits.put("F", FORWARD);
                positionLimits.put("C", CENTER);

                // Must-have players (same as earlier today)
                List<String> mustHaveEntries = Arrays.asList(

                        //"S. De Larrea|G",
                        "N. Hifi|G",
                        //"T. Dorsey|G",
                        "K. Nunn|G",
                        //"T. Blatt|G",
                        //"A. Obst|G",
                        //"E. Osmani|F",
                        //"L. Birutis|C",
                        "D. Motiejunas|C",
                        //"N. Milutinov|C",
                        //"J. Nwora|F",
                        //"N. Reuvers|F",
                        //"M. Birsen|F",
                        //"T. Luwawu-cabarrot|F",
                        "E. Jackson|F",
                        "S. Vezenkov|F"
                );

                List<PlayerVer2> fixedPlayers = new ArrayList<>();
                for (String entry : mustHaveEntries) {
                    String[] parts = entry.split("\\|");
                    if (parts.length != 2) continue;
                    String name = parts[0].trim();
                    String pos = parts[1].trim().toUpperCase();
                    players.stream()
                            .filter(p -> p.getName().equalsIgnoreCase(name) && p.getPos().equalsIgnoreCase(pos))
                            .findFirst()
                            .ifPresent(fixedPlayers::add);
                }

                List<PlayerVer2> bestTeam = buildBestTeam(players, positionLimits, MAX_COST, fixedPlayers);

                double totalCost = bestTeam.stream().mapToDouble(PlayerVer2::getCost).sum();
                double totalPoints = bestTeam.stream().mapToDouble(p -> p.predictedNextFantasyPoints).sum();

                System.out.println("\nOptimal Team (Most Predicted FP for Budget):");
                bestTeam.forEach(System.out::println);
                System.out.printf("%nTotal cost: %.2f / %.2f | Total predicted FP: %.2f%n",
                        totalCost, MAX_COST, totalPoints);

                saveTeamToFile(bestTeam, totalCost, totalPoints, OUTPUTFILENAME);

            } else if (mode == 3) {
                // Upgrade team using optimal_team_ver2.txt as the current team,
                // making up to 4 substitutions to maximize predicted FP.
                Map<String, Integer> positionLimits = new HashMap<>();
                positionLimits.put("G", GUARD);
                positionLimits.put("F", FORWARD);
                positionLimits.put("C", CENTER);

                int targetRoster = rosterSize(positionLimits);

                // Load current team from the file
                List<PlayerVer2> baseTeam = loadTeamFromFile(OUTPUTFILENAME, players);
                if (baseTeam.size() != targetRoster) {
                    System.out.println("Warning: loaded " + baseTeam.size() + " players from '" + OUTPUTFILENAME +
                            "', expected " + targetRoster + ". Proceeding anyway.");
                }

                // Try all substitution counts from 0..4, pick the best
                int maxSubs = 4;
                List<PlayerVer2> bestUpgrade = new ArrayList<>(baseTeam);
                double bestPts = totalPoints(bestUpgrade);

                for (int subs = 0; subs <= maxSubs; subs++) {
                    int keepCount = Math.max(0, targetRoster - subs);
                    keepCount = Math.min(keepCount, baseTeam.size());
                    List<PlayerVer2> candidate = tryWithKeepCount(players, positionLimits, MAX_COST, baseTeam, keepCount);
                    if (candidate.isEmpty()) continue;
                    double pts = totalPoints(candidate);
                    if (pts > bestPts) {
                        bestPts = pts;
                        bestUpgrade = candidate;
                    }
                }

                // Print & save
                System.out.println("\nUpgraded Team (≤ 4 substitutions, maximize next week FP):");
                bestUpgrade.forEach(System.out::println);
                System.out.printf("%nTotal cost: %.2f / %.2f | Total predicted FP: %.2f%n",
                        totalCost(bestUpgrade), MAX_COST, totalPoints(bestUpgrade));

                saveUpgradedTeamToFile(bestUpgrade, baseTeam, OUTPUTFILENAME_WITH_SUBS);

            } else {
                System.out.println("Invalid mode.");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
