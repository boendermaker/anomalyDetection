import express from 'express';
import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import { fileURLToPath } from 'url';
import { ModelTrainer } from './trainmodel.class.js';
import { DataGenerator } from './datagenerator.class.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3000;
const MODEL_DIR = './models';
const EXCLUDED_KEYS = ['timestamp', 'label', 'bulletSettings'];
const modelTrainer = new ModelTrainer();

modelTrainer.setConfig({ modelPath: MODEL_DIR, epochs: 300, excludedKeys: EXCLUDED_KEYS });

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));


app.get('/api/rndgenerate', (req, res) => {
    // Beispiel-Stages mit mehreren variablen Achsen
    const baseLoad = 5.0;
    const totalSteps = 200;
    const getRandom = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    const endAnlauf = getRandom(30, 60);         // Anlauf endet z.B. bei 45
    const endBetrieb = getRandom(140, 180);      // Betrieb endet z.B. bei 165

    const stages = [
        {
            // PHASE 1: ANLAUF (Immer ab 0)
            from: 0,
            to: endAnlauf,
            callback: (local) => ({
                values: { current: 18 * Math.exp(-local / 6) + baseLoad },
                isAnomaly: false
            })
        },
        {
            // PHASE 2: BETRIEB (Startet exakt 1 Schritt nach Anlauf)
            from: endAnlauf + 1,
            to: endBetrieb,
            callback: (local, global) => {
                let val = baseLoad;
                let anomaly = false;

                // Fehler beginnt dynamisch 20 Schritte nach Anlauf-Ende
                const failureStart = endAnlauf + 20;
                if (global > failureStart) {
                    val += (global - failureStart) * 0.12;
                    if (val > 6.5) anomaly = true;
                }

                if (Math.random() > 0.96) val += 5 + Math.random() * 5;

                return { values: { current: val }, isAnomaly: anomaly };
            }
        },
        {
            // PHASE 3: AUSLAUF (Startet exakt 1 Schritt nach Betrieb)
            from: endBetrieb + 1,
            to: totalSteps,
            callback: (local) => ({
                values: { current: baseLoad * Math.exp(-local / 5) },
                isAnomaly: false
            })
        }
    ];

    const generator = new DataGenerator(totalSteps, stages);
    res.json({ success: true, data: generator.generate() });
});


// Endpunkt: Daten generieren
app.get('/api/generate', (req, res) => {
    // Beispiel-Stages mit mehreren variablen Achsen
    const baseLoad = 5.0;
    const stages = [
        {
            // PHASE 1: ANLAUF
            from: 0, to: 50,
            callback: (local) => ({
                values: { current: 18 * Math.exp(-local / 6) + baseLoad },
                isAnomaly: false
            })
        },
        {
            // PHASE 2: BETRIEB & LASTSPITZEN
            from: 51, to: 155,
            callback: (local, global) => {
                let val = baseLoad;
                let anomaly = false;

                // Schleichender Lagerschaden ab Schritt 100
                if (global > 80) {
                    val += (global - 80) * 0.12;
                    if (val > 6.5) anomaly = true; // Ab 8.5A als Fehler markiert
                }

                // Zufällige mechanische Lastspitzen (Spikes)
                if (Math.random() > 0.96) {
                    val += 5 + Math.random() * 5;
                }

                return { values: { current: val }, isAnomaly: anomaly };
            }
        },
        {
            // PHASE 3: AUSLAUF (Coast-down)
            from: 151, to: 200,
            callback: (local) => ({
                values: { current: baseLoad * Math.exp(-local / 5) },
                isAnomaly: false
            })
        }
    ];

    const generator = new DataGenerator(200, stages);
    res.json({ success: true, data: generator.generate() });
});

// Endpunkt: Training
app.post('/api/train', async (req, res) => {
    try {
        const { rawData } = req.body;
        modelTrainer.setRawData(rawData);
        await modelTrainer.train();
        res.json({ success: true, message: 'Generisches Modell trainiert.' });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Endpunkt: Analyse (Predict)
app.post('/api/analyze', async (req, res) => {
    try {
        const { currentData } = req.body;
        const model = await modelTrainer.load();

        if (!model) return res.status(500).json({ error: 'Kein Modell gefunden.' });

        // Automatische Feature-Extraktion für den Tensor
        const featureKeys = Object.keys(currentData[0]).filter(k => !EXCLUDED_KEYS.includes(k));
        const inputMatrix = currentData.map(point => featureKeys.map(k => point[k]));

        const xs = tf.tensor2d(inputMatrix, [currentData.length, featureKeys.length]);
        const predictions = await model.predict(xs).data();

        res.json({ success: true, predictions: Array.from(predictions) });
        xs.dispose();
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, '0.0.0.0', () => console.log(`Server auf Port ${PORT}`));