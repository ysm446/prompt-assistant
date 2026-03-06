'use strict';

const { app, BrowserWindow, dialog, Menu } = require('electron');
const path = require('path');
const http = require('http');

const PORT = 8765;
const SERVER_URL = `http://127.0.0.1:${PORT}`;

let mainWindow = null;

// ---------------------------------------------------------------------------
// Python サーバーが起動するまで待機
// (サーバーは start.bat が conda activate main 後に起動済み)
// ---------------------------------------------------------------------------

function waitForServer(retries = 60) {
  return new Promise((resolve, reject) => {
    const check = remaining => {
      if (remaining <= 0) {
        reject(new Error('Server did not start within 60 seconds.'));
        return;
      }
      const req = http.get(`${SERVER_URL}/api/settings`, res => {
        if (res.statusCode === 200) {
          resolve();
        } else {
          setTimeout(() => check(remaining - 1), 1000);
        }
        res.resume();
      });
      req.on('error', () => setTimeout(() => check(remaining - 1), 1000));
      req.setTimeout(800, () => {
        req.destroy();
        setTimeout(() => check(remaining - 1), 500);
      });
    };
    check(retries);
  });
}

// ---------------------------------------------------------------------------
// ウィンドウ作成
// ---------------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1200,
    minWidth: 900,
    minHeight: 600,
    title: 'Prompt Assistant',
    backgroundColor: '#1c1c1c',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadURL(SERVER_URL);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ---------------------------------------------------------------------------
// アプリライフサイクル
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  // メニューバーを完全に削除
  Menu.setApplicationMenu(null);

  // ディスクキャッシュを無効化（開発中のデザイン変更を即時反映）
  app.commandLine.appendSwitch('disable-http-cache');
  try {
    await waitForServer();
  } catch (err) {
    dialog.showErrorBox(
      'Server not found',
      `Could not connect to Python server at ${SERVER_URL}.\n\nMake sure to launch via start.bat which starts the server first.\n\n${err.message}`
    );
    app.quit();
    return;
  }

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  // Python サーバーにシャットダウンを通知
  http.request({ host: '127.0.0.1', port: PORT, path: '/api/shutdown', method: 'POST' }, () => {}).end();
  if (process.platform !== 'darwin') app.quit();
});
