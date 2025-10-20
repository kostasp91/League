package Ver2;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.json.JSONArray;
import org.json.JSONObject;

public class PlayerRecommenderVer2 {

    private static final String PREDICT_URL = "http://127.0.0.1:5000/predict";
    private static final String XLFILE = "players_data_dunkest.xlsx";
    private static final int TOP_PLAYERS = 30;

    public static class Player {
        public String name, pos, team;
        public double cost;
        public double minutes;          // raw minutes from Excel
        public double expectedMinutes;  // from server (fallback to minutes)
        public double avgFPT, lastFPT, predictedFPT;
        public int matchesPlayed;
        public int matchesTotal;
        public double pPlay = 1.0;      // probability to play (0..1)

        public Player(String name, String pos, String team,
                      double cost, double minutes, double avgFPT, double lastFPT) {
            this.name = name;
            this.pos = pos;
            this.team = team;
            this.cost = cost;
            this.minutes = minutes;
            this.expectedMinutes = minutes;
            this.avgFPT = avgFPT;
            this.lastFPT = lastFPT;
            this.matchesTotal = 1;
            this.matchesPlayed = (minutes > 0.0 ? 1 : 0);
        }

        @Override
        public String toString() {
            return String.format(
                    "%s | %s | Cost: %.2f | AvgFP: %.2f | LastFP: %.2f | PredNextFP: %.2f | ExpMin: %.1f | PPlay: %.2f | %d/%d",
                    name, pos, cost, avgFPT, lastFPT, predictedFPT, expectedMinutes, pPlay, matchesPlayed, matchesTotal
            );
        }
    }

    public static List<Player> loadPlayers() throws IOException {
        List<Player> players = new ArrayList<>();
        Map<String, Player> playerMap = new HashMap<>();
        DataFormatter fmt = new DataFormatter(Locale.ROOT);

        try (FileInputStream fis = new FileInputStream(XLFILE);
             Workbook workbook = new XSSFWorkbook(fis)) {

            // --- Identify latest week ---
            int latestWeekIndex = -1;
            int maxWeekNum = -1;

            for (int i = 0; i < workbook.getNumberOfSheets(); i++) {
                String sheetName = workbook.getSheetName(i);
                if (sheetName != null && sheetName.toLowerCase(Locale.ROOT).startsWith("week")) {
                    try {
                        int num = Integer.parseInt(sheetName.replaceAll("\\D", ""));
                        if (num > maxWeekNum) {
                            maxWeekNum = num;
                            latestWeekIndex = i;
                        }
                    } catch (Exception ignore) {}
                }
            }
            if (latestWeekIndex < 0) {
                // fallback to the last sheet if "Week##" not found
                latestWeekIndex = Math.max(0, workbook.getNumberOfSheets() - 1);
            }

            // --- For the latest sheet, detect the PLUS column from the header ---
            int plusColLatest = -1;
            {
                Sheet latestSheet = workbook.getSheetAt(latestWeekIndex);
                if (latestSheet != null) {
                    Row header = latestSheet.getRow(latestSheet.getFirstRowNum());
                    if (header != null) {
                        for (Cell hc : header) {
                            if (hc == null) continue;
                            String h = fmt.formatCellValue(hc).trim().toLowerCase(Locale.ROOT);
                            // common synonyms for the "price change" column
                            if (h.equals("plus") || h.equals("Δ") || h.contains("plus") || h.contains("delta") || h.contains("change")) {
                                plusColLatest = hc.getColumnIndex();
                                break;
                            }
                        }
                    }
                }
            }

            // --- Process all weeks for performance average ---
            for (int i = 0; i < workbook.getNumberOfSheets(); i++) {
                Sheet sheet = workbook.getSheetAt(i);
                if (sheet == null) continue;

                Iterator<Row> rowIterator = sheet.iterator();
                if (rowIterator.hasNext()) rowIterator.next(); // skip header row

                while (rowIterator.hasNext()) {
                    Row row = rowIterator.next();
                    try {
                        // ORIGINAL fixed indices (keep as-is)
                        String name = row.getCell(0).getStringCellValue();
                        String pos  = row.getCell(1).getStringCellValue();
                        String team = row.getCell(2).getStringCellValue();
                        double fpt  = row.getCell(3).getNumericCellValue();
                        double cr   = row.getCell(4).getNumericCellValue();
                        // NOTE: if your file moved MIN, adjust this index accordingly
                        double minutes = row.getCell(7).getNumericCellValue(); // column index for MIN

                        // Participation tweak (optional): <1 minute counts as DNP
                        if (minutes < 1.0) minutes = 0.0;

                        String key = (name + "|" + pos).toLowerCase(Locale.ROOT);

                        Player p = playerMap.getOrDefault(key,
                                new Player(name, pos, team, cr, minutes, fpt, fpt));

                        // Update averages / last / participation
                        p.avgFPT = (p.avgFPT * p.matchesTotal + fpt) / (p.matchesTotal + 1);
                        p.lastFPT = fpt;
                        p.minutes = minutes;
                        p.matchesTotal++;
                        if (minutes > 0) p.matchesPlayed++;

                        // If this sheet is the latest week, update the latest cost as CR + PLUS (if PLUS exists)
                        if (i == latestWeekIndex) {
                            double plus = 0.0;
                            if (plusColLatest >= 0) {
                                Cell pc = row.getCell(plusColLatest);
                                if (pc != null) {
                                    try {
                                        // robust parse with DataFormatter in case it's a string
                                        String s = fmt.formatCellValue(pc).trim().replace(",", ".");
                                        if (!s.isEmpty()) plus = Double.parseDouble(s);
                                    } catch (Exception ignore) {}
                                }
                            }
                            p.cost = cr + plus; // <<< THE KEY CHANGE
                        }

                        playerMap.put(key, p);
                    } catch (Exception ignore) {}
                }
            }
        }

        players.addAll(playerMap.values());
        return players;
    }

    public static void fetchPredictions(List<Player> players) throws IOException {
        JSONArray playerArray = new JSONArray();
        for (Player p : players) {
            JSONObject obj = new JSONObject();
            obj.put("Player", p.name);
            obj.put("Pos", p.pos);
            obj.put("Team", p.team);
            obj.put("CR", p.cost);          // we already set CR = CR + PLUS for latest week
            obj.put("Minutes", p.minutes);
            obj.put("AvgFP", p.avgFPT);
            obj.put("lastFPT", p.lastFPT);
            playerArray.put(obj);
        }

        JSONObject body = new JSONObject();
        body.put("players", playerArray);

        URL url = new URL(PREDICT_URL);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.toString().getBytes(StandardCharsets.UTF_8));
        }

        StringBuilder response = new StringBuilder();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) response.append(line.trim());
        }

        JSONArray predictions = new JSONArray(response.toString());
        int serverAppliedCR = 0;
        for (int i = 0; i < predictions.length(); i++) {
            JSONObject pred = predictions.getJSONObject(i);
            String name = pred.optString("Player");
            String pos  = pred.optString("Pos");

            for (Player p : players) {
                if (p.name.equalsIgnoreCase(name) && p.pos.equalsIgnoreCase(pos)) {
                    p.predictedFPT    = pred.optDouble("PredictedNextFP", p.predictedFPT);
                    // IMPORTANT: do NOT override cost here, we keep Excel CR+PLUS
                    // double serverCR = pred.optDouble("CR", p.cost);
                    // if (Math.abs(serverCR - p.cost) > 1e-9) serverAppliedCR++;

                    p.expectedMinutes = pred.optDouble("ExpectedMIN", p.minutes);
                    p.pPlay           = pred.optDouble("PPlay", p.pPlay);
                    break;
                }
            }
        }

        // If you want to see if the server was trying to override CR, uncomment the counter above
        // System.out.println("fetchPredictions: server CR applied for " + serverAppliedCR + " players; Excel CR+PLUS kept.");
    }

    public static void main(String[] args) {
        try {
            System.out.println("Loading players...");
            List<Player> players = loadPlayers();

            System.out.println("Requesting AI predictions from Python model...");
            fetchPredictions(players);

            players.sort(Comparator.comparingDouble(p -> -p.predictedFPT));
            System.out.println("\nTop Recommended Players (by predicted fantasy points):");
            for (int i = 0; i < Math.min(TOP_PLAYERS, players.size()); i++) {
                System.out.println(players.get(i));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
