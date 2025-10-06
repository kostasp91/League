

class Player {
    String name, team, pos;
    double avgFantasyPoints, fantasyPointsProgress, cost,  minutesProgress, trueAverageFantasyPoints;
    int minutes, count, matches;

    public Player(String name, String team, String pos, double avgFantasyPoints, double fantasyPointsProgress, double cost, int minutes, double minutesProgress, double trueAverageFantasyPoints, int count, int matches) {
        this.name = name;
        this.team = team;
        this.pos = pos;
        this.avgFantasyPoints = avgFantasyPoints;
        this.fantasyPointsProgress = fantasyPointsProgress;
        this.cost = cost;
        this.minutes = minutes;
        this.minutesProgress = minutesProgress;
        this.trueAverageFantasyPoints = trueAverageFantasyPoints;
        this.count = count;
        this.matches = matches;
    }

    public double getScore() {
        if (count >= 2) {

            double avgFantasyPointsRatio = cost > 0 ? avgFantasyPoints / cost : 0;
            double fantasyProgressRatio = fantasyPointsProgress; //avgFantasyPoints > 0 ? fantasyPointsProgress / avgFantasyPoints : 0;
            double minutesFactor = minutes / 40.0;
            double minutesProgressRatio = minutes > 0 ? minutesProgress / minutes : 0;
            double consistency = matches > 0 ? (double) count / matches : 0.0;

            return avgFantasyPointsRatio * 0.65
                    + fantasyProgressRatio * 0.1
                    + minutesFactor * 0.1
                    + minutesProgressRatio * 0.05
                    + consistency * 0.1;
        } else
            return 0;
    }

    @Override
    public String toString() {
        double consistency = matches > 0 ? (double) count / matches : 0.0;
        return String.format(
                "%s | %s | Games: %d/%d (%.0f%%) | AvgFP: %.2f | Progress: %.2f | Cost: %.2f | AvgMinutes: %d | TrueAvg: %.2f | Score: %.2f",
                name, pos, count, matches, consistency*100, avgFantasyPoints, fantasyPointsProgress, cost, minutes, trueAverageFantasyPoints, getScore()
        );
    }

    public String getPos() {
        return pos;
    }

    public double getCost() {
        return cost;
    }

    public String getName() {
        return name;
    }

    public double getAvgFantasyPoints() {
        return avgFantasyPoints;
    }
}