using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace GestureControlApp
{

    public class LetterMapping : INotifyPropertyChanged
    {
        private string _path = "";
        private string _displayName = "";

        public string Letter { get; set; } = "";

        public string Path
        {
            get => _path;
            set
            {
                _path = value;
                DisplayName = DeriveDisplayName(value);
                OnPropertyChanged(nameof(Path));
            }
        }

        public string DisplayName
        {
            get => _displayName;
            private set { _displayName = value; OnPropertyChanged(nameof(DisplayName)); }
        }

        private static string DeriveDisplayName(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return "—";
            try { return System.IO.Path.GetFileNameWithoutExtension(path); }
            catch { return path; }
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged(string name) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }

    public partial class MainWindow : Window
    {
        private Process? pythonProcess;

        private static readonly Dictionary<string, string> Defaults = new()
        {
            ["A"] = @"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
            ["B"] = "",
            ["C"] = @"C:\Windows\System32\calc.exe",
            ["D"] = @"C:\Windows\explorer.exe",
            ["E"] = @"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            ["F"] = @"C:\Windows\System32\notepad.exe",
            ["G"] = @"C:\Program Files\Google\Chrome\Application\chrome.exe",
            ["H"] = "",
            ["I"] = @"C:\Windows\System32\notepad.exe",
            ["J"] = "",
            ["K"] = @"C:\Windows\System32\notepad.exe",
            ["L"] = @"C:\Windows\explorer.exe",
            ["M"] = "",
            ["N"] = "",
            ["O"] = "",
            ["P"] = @"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
            ["Q"] = "",
            ["R"] = "",
            ["S"] = "",
            ["T"] = @"C:\Windows\System32\cmd.exe",
            ["U"] = @"C:\Windows\System32\notepad.exe",
            ["V"] = @"C:\Program Files\Microsoft VS Code\Code.exe",
            ["W"] = "",
            ["X"] = @"C:\Windows\System32\taskmgr.exe",
            ["Y"] = "",
            ["Z"] = @"C:\Windows\System32\notepad.exe",
        };

        private ObservableCollection<LetterMapping> Mappings { get; } = new();
        private string MappingsFile => System.IO.Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory, "mappings.json");

        public MainWindow()
        {
            InitializeComponent();
            LoadMappings();
            MappingsList.ItemsSource = Mappings;
        }

        private void Tab_Checked(object sender, RoutedEventArgs e)
        {
            if (PanelReference == null || PanelWriting == null) return;

            if (TabReference?.IsChecked == true)
            {
                PanelReference.Visibility = Visibility.Visible;
                PanelWriting.Visibility   = Visibility.Collapsed;
            }
            else
            {
                PanelReference.Visibility = Visibility.Collapsed;
                PanelWriting.Visibility   = Visibility.Visible;
            }
        }

        private void LoadMappings()
        {
            Mappings.Clear();
            Dictionary<string, string>? saved = null;

            if (File.Exists(MappingsFile))
            {
                try
                {
                    var json = File.ReadAllText(MappingsFile);
                    saved = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                }
                catch {  }
            }

            foreach (var letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            {
                var key  = letter.ToString();
                var path = saved != null && saved.TryGetValue(key, out var p) ? p
                         : Defaults.TryGetValue(key, out var d) ? d
                         : "";
                Mappings.Add(new LetterMapping { Letter = key, Path = path });
            }
        }

        private void SaveMappings_Click(object sender, RoutedEventArgs e)
        {
            var dict = new Dictionary<string, string>();
            foreach (var m in Mappings)
                dict[m.Letter] = m.Path;

            try
            {
                var json = JsonSerializer.Serialize(dict,
                    new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(MappingsFile, json);
                SaveHint.Text = $"✓ Saved to mappings.json  —  {DateTime.Now:HH:mm:ss}";
            }
            catch (Exception ex)
            {
                SaveHint.Text = $"✗ Save failed: {ex.Message}";
            }
        }

        private void ResetMappings_Click(object sender, RoutedEventArgs e)
        {
            var r = MessageBox.Show("Reset all mappings to built-in defaults?",
                "Reset Mappings", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (r != MessageBoxResult.Yes) return;

            foreach (var m in Mappings)
                m.Path = Defaults.TryGetValue(m.Letter, out var d) ? d : "";

            SaveHint.Text = "Reset to defaults — press Save Mappings to persist.";
        }

        private void Browse_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not Button btn) return;
            var letter = btn.Tag?.ToString() ?? "";

            var dlg = new OpenFileDialog
            {
                Title            = $"Select app for letter '{letter}'",
                Filter           = "Executables (*.exe)|*.exe|All files (*.*)|*.*",
                CheckFileExists  = true,
            };

            if (dlg.ShowDialog() == true)
            {
                var mapping = FindMapping(letter);
                if (mapping != null) mapping.Path = dlg.FileName;
            }
        }

        private LetterMapping? FindMapping(string letter)
        {
            foreach (var m in Mappings)
                if (m.Letter == letter) return m;
            return null;
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {

            var dict = new Dictionary<string, string>();
            foreach (var m in Mappings)
                dict[m.Letter] = m.Path;
            try
            {
                File.WriteAllText(MappingsFile,
                    JsonSerializer.Serialize(dict, new JsonSerializerOptions { WriteIndented = true }));
            }
            catch {  }

            string baseDir    = AppDomain.CurrentDomain.BaseDirectory;
            string venvPython = System.IO.Path.Combine(baseDir, "venv", "Scripts", "python.exe");
            string script     = System.IO.Path.Combine(baseDir, "gesture_control.py");

            if (!File.Exists(venvPython))
            {
                MessageBox.Show("Python venv not found.\nRun setup.bat first.",
                    "Setup Required", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (!File.Exists(script))
            {
                MessageBox.Show("gesture_control.py not found.",
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
                    CreateNoWindow   = true,
                };
                pythonProcess = Process.Start(psi);

                StartButton.IsEnabled = false;
                StopButton.IsEnabled  = true;
                StatusText.Text       = "Running";
                StatusSub.Text        = "Gesture control is active";
                StatusText.Foreground = System.Windows.Media.Brushes.LightGreen;
                StatusDot.Fill        = new System.Windows.Media.SolidColorBrush(
                                            System.Windows.Media.Color.FromRgb(74, 222, 128));
                HeaderStatus.Text     = "Running";
                HeaderStatus.Foreground = System.Windows.Media.Brushes.LightGreen;
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
            ResetStatus();
        }

        private void KillPython()
        {
            try { pythonProcess?.Kill(entireProcessTree: true); } catch { }
            pythonProcess = null;
        }

        private void ResetStatus()
        {
            StartButton.IsEnabled = true;
            StopButton.IsEnabled  = false;
            StatusText.Text       = "Stopped";
            StatusSub.Text        = "Press Start to run again";
            StatusText.Foreground = new System.Windows.Media.SolidColorBrush(
                                        System.Windows.Media.Color.FromRgb(90, 90, 120));
            StatusDot.Fill        = new System.Windows.Media.SolidColorBrush(
                                        System.Windows.Media.Color.FromRgb(90, 90, 120));
            HeaderStatus.Text     = "Idle";
            HeaderStatus.Foreground = new System.Windows.Media.SolidColorBrush(
                                        System.Windows.Media.Color.FromRgb(90, 90, 120));
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            KillPython();
            base.OnClosing(e);
        }
    }
}
