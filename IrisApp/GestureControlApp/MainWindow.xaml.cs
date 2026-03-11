using System;
using System.Diagnostics;
using System.IO;
using System.Windows;

namespace GestureControlApp
{
    public partial class MainWindow : Window
    {
        private Process? pythonProcess;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {
            string baseDir    = AppDomain.CurrentDomain.BaseDirectory;
            string venvPython = Path.Combine(baseDir, "venv", "Scripts", "python.exe");
            string script     = Path.Combine(baseDir, "gesture_control.py");

            if (!File.Exists(venvPython))
            {
                MessageBox.Show("Python venv not found.\nPlease run setup.bat first.",
                    "Setup Required", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (!File.Exists(script))
            {
                MessageBox.Show("gesture_control.py not found next to the launcher.",
                    "File Missing", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName         = venvPython,
                    Arguments        = $"\"{script}\"",
                    WorkingDirectory = baseDir,
                    UseShellExecute  = false,
                    CreateNoWindow   = true
                };
                pythonProcess = Process.Start(psi);

                StartButton.IsEnabled  = false;
                StopButton.IsEnabled   = true;
                StatusText.Text        = "● Running";
                StatusText.Foreground  = System.Windows.Media.Brushes.Green;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to start:\n{ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            KillPython();
            StartButton.IsEnabled  = true;
            StopButton.IsEnabled   = false;
            StatusText.Text        = "Stopped";
            StatusText.Foreground  = System.Windows.Media.Brushes.Gray;
        }

        private void KillPython()
        {
            try { pythonProcess?.Kill(entireProcessTree: true); } catch { }
            pythonProcess = null;
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            KillPython();
            base.OnClosing(e);
        }
    }
}