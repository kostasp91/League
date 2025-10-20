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
        public double minutes;          // raw minutes from Excel (DNP rule applied)
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
            this.matchesTotal = 1;                   // FIRST observation is counted here
            this.matchesPlayed = (minutes > 0.0 ? 1 : 0);
        }

        @Override
        public String toString() {
            return String.format(
                    "%s | %s | Pred FP: %.2f |  AVG FP: %.2f | Cost: %.2f | ExpMin: %.1f | PPlay: %.2f",
                    name, pos, predictedFPT, avgFPT, cost, expectedMinutes, pPlay
            );
        }
    }

    public static List<Player> loadPlayers() throws IOException {
        List<Player> players = new ArrayList<>();
        Map<String, Player> playerMap = new HashMap<>();
        DataFormatter fmt = new DataFormatter(Locale.ROOT);

        try (FileInputStream fis = new FileInputStream(XLFILE);
             Workbook workbook = new XSSFWorkbook(fis)) {

            // --- Identify latest week sheet (prefer "Week##", else last sheet) ---
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
                latestWeekIndex = Math.max(0, workbook.getNumberOfSheets() - 1);
            }

            // --- Detect PLUS column in the latest sheet by header name ---
            int plusColLatest = -1;
            Sheet latestSheet = workbook.getSheetAt(latestWeekIndex);
            if (latestSheet != null) {
                Row header = latestSheet.getRow(latestSheet.getFirstRowNum());
                if (header != null) {
                    for (Cell hc : header) {
                        if (hc == null) continue;
                        String h = fmt.formatCellValue(hc).trim().toLowerCase(Locale.ROOT);
                        // loose matching: "plus", "delta", "change", etc.
                        if (h.equals("plus") || h.contains("plus") || h.contains("delta") || h.contains("change")) {
                            plusColLatest = hc.getColumnIndex();
                            break;
                        }
                    }
                }
            }

            // --- Process all weeks; fixed column indices like your original code ---
            for (int i = 0; i < workbook.getNumberOfSheets(); i++) {
                Sheet sheet = workbook.getSheetAt(i);
                if (sheet == null) continue;

                Iterator<Row> rowIterator = sheet.iterator();
                if (rowIterator.hasNext()) rowIterator.next(); // skip header

                while (rowIterator.hasNext()) {
                    Row row = rowIterator.next();
                    try {
                        // ORIGINAL fixed indices (adjust only if your Excel changed!)
                        String name = row.getCell(0).getStringCellValue();
                        String pos  = row.getCell(1).getStringCellValue();
                        String team = row.getCell(2).getStringCellValue();
                        double fpt  = row.getCell(3).getNumericCellValue();
                        double cr   = row.getCell(4).getNumericCellValue();
                        double minutesRaw = row.getCell(7).getNumericCellValue(); // MIN

                        // Participation rule: < 1 minute counts as DNP
                        double minutes = (minutesRaw >= 1.0) ? minutesRaw : 0.0;

                        String key = (name + "|" + pos).toLowerCase(Locale.ROOT);
                        Player p = playerMap.get(key);

                        if (p == null) {
                            // FIRST observation → set baseline ONLY (do NOT apply running average yet)
                            p = new Player(name, pos, team, cr, minutes, fpt, fpt);

                            // If we are on the latest sheet, set CR = CR + PLUS (if PLUS detected)
                            if (i == latestWeekIndex) {
                                double plus = 0.0;
                                if (plusColLatest >= 0) {
                                    Cell pc = row.getCell(plusColLatest);
                                    if (pc != null) {
                                        try {
                                            String s = fmt.formatCellValue(pc).trim().replace(",", ".");
                                            if (!s.isEmpty()) plus = Double.parseDouble(s);
                                        } catch (Exception ignore) {}
                                    }
                                }
                                p.cost = cr + plus; // << key requirement
                            }
                            playerMap.put(key, p);
                        } else {
                            // SUBSEQUENT observations → apply correct running arithmetic mean
                            p.avgFPT = (p.avgFPT * p.matchesTotal + fpt) / (p.matchesTotal + 1);
                            p.matchesTotal++;
                            if (minutes > 0) p.matchesPlayed++;
                            p.lastFPT = fpt;
                            p.minutes = minutes;
                            p.expectedMinutes = minutes;

                            // Update CR to CR+PLUS if this row is on the latest week
                            if (i == latestWeekIndex) {
                                double plus = 0.0;
                                if (plusColLatest >= 0) {
                                    Cell pc = row.getCell(plusColLatest);
                                    if (pc != null) {
                                        try {
                                            String s = fmt.formatCellValue(pc).trim().replace(",", ".");
                                            if (!s.isEmpty()) plus = Double.parseDouble(s);
                                        } catch (Exception ignore) {}
                                    }
                                }
                                p.cost = cr + plus; // latest price = CR + PLUS
                            }
                        }
                    } catch (Exception ignore) {
                        // swallow & continue: malformed row
                    }
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
            obj.put("CR", p.cost);              // keep Excel CR+PLUS
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
        for (int i = 0; i < predictions.length(); i++) {
            JSONObject pred = predictions.getJSONObject(i);
            String name = pred.optString("Player");
            String pos  = pred.optString("Pos");

            for (Player p : players) {
                if (p.name.equalsIgnoreCase(name) && p.pos.equalsIgnoreCase(pos)) {
                    p.predictedFPT    = pred.optDouble("PredictedNextFP", p.predictedFPT);
                    // DO NOT override p.cost — we keep Excel CR+PLUS
                    p.expectedMinutes = pred.optDouble("ExpectedMIN", p.minutes);
                    p.pPlay           = pred.optDouble("PPlay", p.pPlay);
                    break;
                }
            }
        }
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
