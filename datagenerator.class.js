export class DataGenerator {

    constructor(totalSteps, stages, intervalMs = 1000) {
        this.totalSteps = totalSteps;
        this.stages = stages;
        this.intervalMs = intervalMs;
        this.startTime = Date.now();
    }

    generate() {
        return Array.from({ length: this.totalSteps }, (_, i) => {
            const timestamp = this.startTime + (i * this.intervalMs);
            const stage = this.stages.find(s => i >= s.from && i <= s.to);

            let result = { values: {}, isAnomaly: false };
            if (stage?.callback) {
                const cbRes = stage.callback(i - stage.from, i);
                result = typeof cbRes === 'object' ? cbRes : { values: { val: cbRes }, isAnomaly: false };
            }

            // Rauschen auf alle numerischen Werte in 'values'
            const processedValues = {};
            for (const [key, val] of Object.entries(result.values || {})) {
                processedValues[key] = Math.max(0, val + (Math.random() - 0.5) * 0.2);
            }

            return {
                timestamp,
                step: i,
                ...processedValues,
                label: result.isAnomaly ? 1 : 0
            };
        });
    }

}