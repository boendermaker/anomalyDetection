import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';

export class ModelTrainer {

    constructor() {
        this.modelPath = path.resolve('./models');
        this.epochs = 100;
        this.model = null;
        this.rawData = null;
        // Diese Felder sind Meta-Daten oder Targets, keine Eingangs-Features
        this.excludedKeys = ['timestamp', 'label', 'bulletSettings'];
    }

    /**
     * Ermittelt dynamisch alle verfügbaren Feature-Keys aus dem ersten Datenpunkt
     */
    _getFeatureKeys(sample) {
        return Object.keys(sample).filter(key => !this.excludedKeys.includes(key));
    }

    setConfig(config) {
        if (config.modelPath) this.modelPath = path.resolve(config.modelPath);
        if (config.epochs) this.epochs = config.epochs;
        if (config.excludedKeys) this.excludedKeys = config.excludedKeys;
    }

    setRawData(rawData) {
        this.rawData = rawData;
    }

    _buildModel(inputDim) {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [inputDim] }));
        model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        this.model = model;
    }

    async train() {
        if (!this.rawData.length) throw new Error('Keine Daten.');

        const featureKeys = this._getFeatureKeys(this.rawData[0]);
        const inputDim = featureKeys.length;

        if (!this.model) {
            const jsonPath = path.join(this.modelPath, 'model.json');
            fs.existsSync(jsonPath) ? await this.load() : this._buildModel(inputDim);
        }

        console.log(`--- Training mit ${inputDim} Features: [${featureKeys}] ---`);

        // Generisches Mapping der Matrix
        const xsMatrix = this.rawData.map(point => featureKeys.map(k => point[k]));
        const ysArray = this.rawData.map(d => d.label);

        const xs = tf.tensor2d(xsMatrix, [this.rawData.length, inputDim]);
        const ys = tf.tensor2d(ysArray, [this.rawData.length, 1]);

        await this.model.fit(xs, ys, {
            epochs: this.epochs,
            verbose: 0
        });

        xs.dispose();
        ys.dispose();
        return this.save();
    }

    async load() {
        const loadPath = `file://${this.modelPath}/model.json`;
        try {
            this.model = await tf.loadLayersModel(loadPath);
            // Re-compile mit generischem Loss für Klassifizierung
            this.model.compile({ optimizer: tf.train.adam(0.001), loss: 'binaryCrossentropy' });
            return this.model;
        } catch (e) {
            this.model = null;
        }
    }

    async save() {
        if (!fs.existsSync(this.modelPath)) fs.mkdirSync(this.modelPath, { recursive: true });
        await this.model.save(`file://${this.modelPath}`);
        return true;
    }
}