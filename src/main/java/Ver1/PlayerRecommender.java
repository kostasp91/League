package Ver1;

import java.io.*;
import java.util.*;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class PlayerRecommender {

    private static final double WANTED_COST = 8.4;
    private static final String WANTED_POSITION = "G";
    private static final int MIN_GAMES_PLAYED = 1;
    private static final int TOP_PLAYERS = 30;
    private static final String XLFILE = "players_data_dunkest.xlsx";

    static class Stats {
        String team;
        String pos;
        double fantasyPointsSum = 0;
        double fantasyPointsLast = 0;
        double fantasyPointsSecondToLast = 0;
        double fantasyPointsThirdToLast = 0;
        double fantasyPointsFourthToLast = 0;
        double costSum = 0;
        double latestCost = 0;
        int minutesSum = 0;
        int minutesLast = 0;
        int minutesSecondToLast = 0;
        int minutesThirdToLast = 0;
        int minutesFourthToLast = 0;
        int count = 0;
        int recentWeeksCount = 0;
    }

    public static List<Player> getAllPlayers() throws IOException {
        Map<String, Stats> playerStatsMap = new HashMap<>();
        int weeks;

        try (FileInputStream fis = new FileInputStream(XLFILE);
             Workbook workbook = new XSSFWorkbook(fis)) {

            for (weeks = 0; weeks < workbook.getNumberOfSheets(); weeks++) {
                Sheet sheet = workbook.getSheetAt(weeks);
                Iterator<Row> rowIterator = sheet.iterator();

                if (rowIterator.hasNext()) rowIterator.next(); // skip header

                while (rowIterator.hasNext()) {
                    Row row = rowIterator.next();
                    try {
                        String name = row.getCell(0).getStringCellValue();
                        String pos = row.getCell(1).getStringCellValue();
                        String team = row.getCell(2).getStringCellValue();
                        double fantasyPoints = row.getCell(3).getNumericCellValue();
                        double cost = row.getCell(4).getNumericCellValue();
                        int minutes = (int) row.getCell(5).getNumericCellValue();

                        // FIX: use composite key
                        String key = name + "|" + pos + "|" + team;
                        Stats stats = playerStatsMap.getOrDefault(key, new Stats());
                        stats.pos = pos;
                        stats.team = team;
                        stats.fantasyPointsSum += fantasyPoints;
                        stats.costSum += cost;
                        stats.minutesSum += minutes;
                        stats.count++;
                        stats.latestCost = cost;

                        int totalSheets = workbook.getNumberOfSheets();
                        if (weeks == totalSheets - 1) {
                            stats.fantasyPointsLast = fantasyPoints;
                            stats.minutesLast = minutes;
                            stats.recentWeeksCount++;
                        }
                        if (weeks == totalSheets - 2 && totalSheets - 2 >= 0) {
                            stats.fantasyPointsSecondToLast = fantasyPoints;
                            stats.minutesSecondToLast = minutes;
                            stats.recentWeeksCount++;
                        }
                        if (weeks == totalSheets - 3 && totalSheets - 3 >= 0) {
                            stats.fantasyPointsThirdToLast = fantasyPoints;
                            stats.minutesThirdToLast = minutes;
                            stats.recentWeeksCount++;
                        }
                        if (weeks == totalSheets - 4 && totalSheets - 4 >= 0) {
                            stats.fantasyPointsFourthToLast = fantasyPoints;
                            stats.minutesFourthToLast = minutes;
                            stats.recentWeeksCount++;
                        }

                        playerStatsMap.put(key, stats);
                    } catch (Exception e) {
                        // ignore malformed rows
                    }
                }
            }
        }

        List<Player> players = new ArrayList<>();
        for (Map.Entry<String, Stats> entry : playerStatsMap.entrySet()) {
            String[] keyParts = entry.getKey().split("\\|");
            String name = keyParts[0];
            String pos = keyParts[1];
            Stats s = entry.getValue();
            String team = s.team;

            double recentFantasySum = 0;
            double recentMinutesSum = 0;
            int divisor = s.recentWeeksCount > 0 ? s.recentWeeksCount : 1;

            ///// TO DO /////
            recentFantasySum += s.fantasyPointsLast;
            if (divisor >= 2) recentFantasySum += s.fantasyPointsSecondToLast;
            if (divisor >= 3) recentFantasySum += s.fantasyPointsThirdToLast;
            if (divisor >= 4) recentFantasySum += s.fantasyPointsFourthToLast;

            recentMinutesSum += s.minutesLast;
            if (divisor >= 2) recentMinutesSum += s.minutesSecondToLast;
            if (divisor >= 3) recentMinutesSum += s.minutesThirdToLast;
            if (divisor >= 4) recentMinutesSum += s.minutesFourthToLast;
            ////////////////
            double fantasyPointsProgress = s.fantasyPointsSecondToLast != 0
                    ? (s.fantasyPointsLast - s.fantasyPointsSecondToLast) / Math.abs(s.fantasyPointsSecondToLast)
                    : 0;

            double minutesProgress = s.minutesSecondToLast != 0
                    ? (s.minutesLast - s.minutesSecondToLast) / s.minutesSecondToLast
                    : 0;

            double avgFantasyPoints = s.count > 0 ? s.fantasyPointsSum / s.count : 0;
            double avgCost = s.count > 0 ? s.costSum / s.count : 0;
            int avgMinutes = (int) Math.round((double) s.minutesSum / s.count);
            double cost = s.latestCost;
            double trueAverageFantasyPoints = weeks > 0 ? s.fantasyPointsSum / weeks : 0;

            players.add(new Player(name, team, pos, avgFantasyPoints, fantasyPointsProgress, avgCost, cost,
                    avgMinutes, minutesProgress, trueAverageFantasyPoints, s.count, weeks));
        }

        players.sort(Comparator.comparingDouble(Player::getScore).reversed());
        return players;
    }

    public static void main(String[] args) throws IOException {
        List<Player> players = getAllPlayers();
        System.out.println("Top recommended players:");
        for (int i = 0; i < Math.min(TOP_PLAYERS, players.size()); i++) {
            System.out.println(players.get(i));
        }
    }
}
