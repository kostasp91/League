package Ver1;

import java.util.Objects;

class Player {
    String name, team, pos;
    double avgFantasyPoints, fantasyPointsProgress, avgCost, latestCost, minutesProgress, trueAverageFantasyPoints;
    int avgMinutes, count, matches;

    public Player(String name, String team, String pos,
                  double avgFantasyPoints, double fantasyPointsProgress, double avgCost, double latestCost,
                  int avgMinutes, double minutesProgress, double trueAverageFantasyPoints,
                  int count, int matches) {
        this.name = name;
        this.team = team;
        this.pos = pos;
        this.avgFantasyPoints = avgFantasyPoints;
        this.fantasyPointsProgress = fantasyPointsProgress;
        this.avgCost = avgCost;
        this.latestCost = latestCost;
        this.avgMinutes = avgMinutes;
        this.minutesProgress = minutesProgress;
        this.trueAverageFantasyPoints = trueAverageFantasyPoints;
        this.count = count;
        this.matches = matches;
    }

    public double getScore() {

        //if (latestCost <= 10.7) {

            double avgFantasyPointsRatio = avgCost > 0 ? avgFantasyPoints / avgCost : 0;
            double fantasyProgressRatio = fantasyPointsProgress;
            double avgMinutesFactor = avgMinutes / 40.0;
            double minutesProgressRatio = minutesProgress;
            double avgFantasyPerMinute = avgMinutes > 0 ? avgFantasyPoints / avgMinutes : 0;
            double participation = matches > 0 ? (double) count / matches : 0.0;

            return (avgFantasyPointsRatio * 0.75
            //        + fantasyProgressRatio * 0.1
                    + avgMinutesFactor * 0.1
                    + minutesProgressRatio * 0.05)
                    * participation;
        //} else
        //    return 0;

    }

    @Override
    public String toString() {
        double participation = matches > 0 ? (double) count / matches : 0.0;
        return String.format(
                "%s | %s | Games: %d/%d (%.0f%%) | AvgFP: %.2f | Progress: %.2f | Average Cost: %.2f | Latest Cost: %.2f | AvgMinutes: %d | Score: %.2f",
                name, pos, count, matches, participation * 100, avgFantasyPoints, fantasyPointsProgress, avgCost, latestCost, avgMinutes, getScore()
        );
    }

    public String getPos() {
        return pos;
    }

    public double getavgCost() {
        return avgCost;
    }

    public double getCost() {
        return latestCost;
    }

    public String getName() {
        return name;
    }

    public double getAvgFantasyPoints() {
        return avgFantasyPoints;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Player other = (Player) obj;
        return name.equalsIgnoreCase(other.name) && pos.equalsIgnoreCase(other.pos);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name.toLowerCase(), pos.toLowerCase());
    }
}
