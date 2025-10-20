package Ver2;

import java.util.Objects;

class PlayerVer2 {
    String name, team, pos;
    double avgFantasyPoints, predictedNextFantasyPoints;
    double latestCost;
    double expectedMinutes; // from server or Excel
    int matchesPlayed;
    int matchesTotal;
    double pPlay;           // 0..1 probability to play

    public PlayerVer2(String name, String team, String pos,
                      double avgFantasyPoints, double predictedNextFantasyPoints, double latestCost,
                      double expectedMinutes, int matchesPlayed, int matchesTotal, double pPlay) {
        this.name = name;
        this.team = team;
        this.pos = pos;
        this.predictedNextFantasyPoints = predictedNextFantasyPoints;
        this.avgFantasyPoints = avgFantasyPoints;
        this.latestCost = latestCost;
        this.expectedMinutes = expectedMinutes;
        this.matchesPlayed = matchesPlayed;
        this.matchesTotal = matchesTotal;
        this.pPlay = pPlay;
    }

    private static double minutesWeight(double m) {
        // Softer tie-breaker among likely-to-play players
        return 1.0 / (1.0 + Math.exp(-(m - 18.0) / 4.0));
    }

    public double getEfficiency() {
        double base = latestCost > 0 ? predictedNextFantasyPoints / latestCost : 0.0;
        // Strong weight on availability; minutes as soft tie-breaker
        double availabilityWeight = Math.pow(Math.max(0.0, Math.min(1.0, pPlay)), 1.6);
        double minuteWeight       = Math.pow(minutesWeight(this.expectedMinutes), 0.5);
        return base * availabilityWeight * minuteWeight;
    }

    public double getAvgFantasyPoints() { return avgFantasyPoints; }
    public double getCost() { return latestCost; }
    public String getPos() { return pos; }
    public String getName() { return name; }

    @Override
    public String toString() {
        return String.format("%s | %s | Pred FP: %.2f |  AVG FP: %.2f | Cost: %.2f | ExpMin: %.1f | PPlay: %.2f | Eff: %.3f",
                name, pos, predictedNextFantasyPoints, avgFantasyPoints, latestCost, expectedMinutes, pPlay, getEfficiency());
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        PlayerVer2 other = (PlayerVer2) obj;
        return name.equalsIgnoreCase(other.name) && pos.equalsIgnoreCase(other.pos);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name.toLowerCase(), pos.toLowerCase());
    }
}
