import express from 'express';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Helper to run python pipeline
function runPythonPipeline(payload: object): Promise<any> {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', [path.join(__dirname, 'run_pipeline.py')]);
    let stdoutData = '';
    let stderrData = '';

    py.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    py.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    py.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(stderrData || `Python process exited with code ${code}`));
      }
      try {
        const parsed = JSON.parse(stdoutData);
        resolve(parsed);
      } catch (err) {
        reject(new Error(`Failed to parse Python output: ${stdoutData.slice(0, 300)}`));
      }
    });

    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();
  });
}

// API Routes
app.post('/api/process', async (req, res) => {
  try {
    const result = await runPythonPipeline({ action: 'process', ...req.body });
    res.json(result);
  } catch (err: any) {
    console.error('Processing error:', err);
    res.status(500).json({ status: 'error', message: err.message });
  }
});

app.post('/api/benchmark', async (req, res) => {
  try {
    const result = await runPythonPipeline({ action: 'benchmark', ...req.body });
    res.json(result);
  } catch (err: any) {
    console.error('Benchmark error:', err);
    res.status(500).json({ status: 'error', message: err.message });
  }
});

app.get('/api/samples', (req, res) => {
  const sampleDir = path.join(__dirname, 'sample_images');
  if (!fs.existsSync(sampleDir)) {
    return res.json([]);
  }
  const files = fs.readdirSync(sampleDir).filter(f => f.endsWith('.png') || f.endsWith('.jpg'));
  res.json(files);
});

app.use('/sample_images', express.static(path.join(__dirname, 'sample_images')));

// Vite middleware in dev or static files in production
async function startServer() {
  if (process.env.NODE_ENV === 'production' && fs.existsSync(path.join(__dirname, 'dist'))) {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  } else {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Edge Detection Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
});
