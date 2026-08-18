using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace SignifyUI.Services
{
    /// <summary>
    /// Backend-based authentication service.
    /// Calls the Python FastAPI account endpoints for registration and login.
    /// </summary>
    public static class AuthService
    {
        public sealed class AccountInfo
        {
            public int Id { get; init; }
            public string Username { get; init; } = string.Empty;
            public string Name { get; init; } = string.Empty;
            public string MasteryLevel { get; init; } = string.Empty;
            public Dictionary<string, int> Progress { get; init; } = new Dictionary<string, int>();
        }

        public sealed class ProgressUpdateResult
        {
            public string Message { get; init; } = string.Empty;
            public string MasteryLevel { get; init; } = string.Empty;
            public Dictionary<string, int> Progress { get; init; } = new Dictionary<string, int>();
        }

        private static readonly HttpClient Http = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(8)
        };

        private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true
        };

        public static string BackendBaseUrl { get; set; } = "http://127.0.0.1:8000";

        private static readonly string LearnProgressRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "SignifyUI",
            "learn_progress");

        /// <summary>Canonical username for the signed-in user, or null when logged out.</summary>
        public static string? CurrentUsername { get; private set; }

        public static bool IsLoggedIn => !string.IsNullOrWhiteSpace(CurrentUsername);

        /// <summary>Fired when the user logs in or out (including <see cref="Logout"/>).</summary>
        public static event EventHandler? SessionChanged;

        static AuthService()
        {
            Http.DefaultRequestHeaders.Accept.Clear();
            Http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }

        /// <summary>
        /// JSON path for the current user's Learn-mode letter progress, or null when not signed in.
        /// </summary>
        public static string? GetLearnProgressFilePath()
        {
            if (string.IsNullOrWhiteSpace(CurrentUsername))
            {
                return null;
            }

            string safe = SanitizeFileNameSegment(CurrentUsername);
            if (string.IsNullOrEmpty(safe))
            {
                return null;
            }

            return Path.Combine(LearnProgressRoot, $"{safe}.json");
        }

        private static string SanitizeFileNameSegment(string name)
        {
            var chars = name.Trim().ToCharArray();
            for (int i = 0; i < chars.Length; i++)
            {
                if (Array.IndexOf(Path.GetInvalidFileNameChars(), chars[i]) >= 0)
                {
                    chars[i] = '_';
                }
            }

            string s = new string(chars);
            return s.Length > 80 ? s[..80] : s;
        }

        public static void Logout()
        {
            if (!IsLoggedIn)
            {
                return;
            }

            CurrentUsername = null;
            SessionChanged?.Invoke(null, EventArgs.Empty);
        }

        private static void SetLoggedInUser(string username)
        {
            CurrentUsername = username;
            SessionChanged?.Invoke(null, EventArgs.Empty);
        }

        /// <summary>
        /// Register a new user with username and password.
        /// Returns success message or error message.
        /// </summary>
        public static (bool success, string message) RegisterUser(string username, string password)
        {
            if (string.IsNullOrWhiteSpace(username))
                return (false, "Username cannot be empty.");

            if (string.IsNullOrWhiteSpace(password))
                return (false, "Password cannot be empty.");

            if (username.Length < 3)
                return (false, "Username must be at least 3 characters.");

            if (password.Length < 6)
                return (false, "Password must be at least 6 characters.");

            var payload = new
            {
                username = username,
                password = password,
                name = username,
                mastery_level = "Beginner"
            };

            return PostJson("/account/create", payload, "Account created successfully! You can now log in.");
        }

        /// <summary>
        /// Authenticate a user with username and password.
        /// Returns success status and message.
        /// </summary>
        public static (bool success, string message) LoginUser(string username, string password)
        {
            if (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(password))
                return (false, "Username and password required.");

            var payload = new
            {
                username = username,
                password = password
            };

            try
            {
                string url = BuildUrl("/account/login");
                string json = JsonSerializer.Serialize(payload, JsonOptions);
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = Http.PostAsync(url, content).GetAwaiter().GetResult();
                string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                if (!response.IsSuccessStatusCode)
                {
                    string detail = ExtractErrorMessage(body);
                    return (false, string.IsNullOrWhiteSpace(detail) ? "Username or password incorrect." : detail);
                }

                using JsonDocument doc = JsonDocument.Parse(body);
                if (
                    doc.RootElement.TryGetProperty("account", out JsonElement accountEl) &&
                    accountEl.TryGetProperty("username", out JsonElement usernameEl)
                )
                {
                    string? canonicalUsername = usernameEl.GetString();
                    if (!string.IsNullOrWhiteSpace(canonicalUsername))
                    {
                        SetLoggedInUser(canonicalUsername);
                        return (true, $"Welcome back, {canonicalUsername}!");
                    }
                }

                // Fallback to input username if backend response shape changes.
                SetLoggedInUser(username);
                return (true, $"Welcome back, {username}!");
            }
            catch (Exception ex)
            {
                return (false, $"Could not reach backend: {ex.Message}");
            }
        }

        /// <summary>
        /// Change the password for a user by calling the backend endpoint.
        /// Returns success status and a message from the backend.
        /// </summary>
        public static (bool success, string message) ChangePassword(string username, string oldPassword, string newPassword)
        {
            if (string.IsNullOrWhiteSpace(username))
                return (false, "Username cannot be empty.");

            if (string.IsNullOrWhiteSpace(oldPassword))
                return (false, "Old password cannot be empty.");

            if (string.IsNullOrWhiteSpace(newPassword))
                return (false, "New password cannot be empty.");

            var payload = new
            {
                username = username,
                old_password = oldPassword,
                new_password = newPassword
            };

            try
            {
                string url = BuildUrl("/account/change_password");
                string json = JsonSerializer.Serialize(payload, JsonOptions);
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = Http.PostAsync(url, content).GetAwaiter().GetResult();
                string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                if (!response.IsSuccessStatusCode)
                {
                    string detail = ExtractErrorMessage(body);
                    return (false, string.IsNullOrWhiteSpace(detail) ? "Failed to change password." : detail);
                }

                // Success
                return (true, "Password changed successfully.");
            }
            catch (Exception ex)
            {
                return (false, $"Could not reach backend: {ex.Message}");
            }
        }

        /// <summary>
        /// Fetches account details for the currently signed-in user.
        /// </summary>
        public static (bool success, string message, AccountInfo? account) GetCurrentAccount()
        {
            if (string.IsNullOrWhiteSpace(CurrentUsername))
            {
                return (false, "No user is currently signed in.", null);
            }

            return GetAccount(CurrentUsername);
        }

        /// <summary>
        /// Fetches account details for a specific username.
        /// </summary>
        public static (bool success, string message, AccountInfo? account) GetAccount(string username)
        {
            if (string.IsNullOrWhiteSpace(username))
            {
                return (false, "Username cannot be empty.", null);
            }

            try
            {
                string url = BuildUrl($"/account/{Uri.EscapeDataString(username)}");
                using var response = Http.GetAsync(url).GetAwaiter().GetResult();
                string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                if (!response.IsSuccessStatusCode)
                {
                    string detail = ExtractErrorMessage(body);
                    return (false, string.IsNullOrWhiteSpace(detail) ? "Failed to fetch account." : detail, null);
                }

                using JsonDocument doc = JsonDocument.Parse(body);
                if (!doc.RootElement.TryGetProperty("account", out JsonElement accountEl))
                {
                    return (false, "Invalid backend response: missing account field.", null);
                }

                AccountInfo info = ParseAccountInfo(accountEl);
                return (true, "OK", info);
            }
            catch (Exception ex)
            {
                return (false, $"Could not reach backend: {ex.Message}", null);
            }
        }

        /// <summary>
        /// Updates progress for the currently signed-in user and returns updated mastery/progress.
        /// </summary>
        public static (bool success, string message, ProgressUpdateResult? result) UpdateProgress(string letter, int level)
        {
            if (string.IsNullOrWhiteSpace(CurrentUsername))
            {
                return (false, "No user is currently signed in.", null);
            }

            if (string.IsNullOrWhiteSpace(letter))
            {
                return (false, "Letter cannot be empty.", null);
            }

            var payload = new
            {
                username = CurrentUsername,
                letter = letter,
                level = level,
            };

            try
            {
                string url = BuildUrl("/account/progress");
                string json = JsonSerializer.Serialize(payload, JsonOptions);
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = Http.PostAsync(url, content).GetAwaiter().GetResult();
                string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                if (!response.IsSuccessStatusCode)
                {
                    string detail = ExtractErrorMessage(body);
                    return (false, string.IsNullOrWhiteSpace(detail) ? "Failed to update progress." : detail, null);
                }

                using JsonDocument doc = JsonDocument.Parse(body);
                string message = doc.RootElement.TryGetProperty("message", out JsonElement messageEl)
                    ? (messageEl.GetString() ?? "Progress updated.")
                    : "Progress updated.";

                string masteryLevel = doc.RootElement.TryGetProperty("mastery_level", out JsonElement masteryEl)
                    ? (masteryEl.GetString() ?? string.Empty)
                    : string.Empty;

                Dictionary<string, int> progress = doc.RootElement.TryGetProperty("progress", out JsonElement progressEl)
                    ? ParseProgress(progressEl)
                    : new Dictionary<string, int>();

                var result = new ProgressUpdateResult
                {
                    Message = message,
                    MasteryLevel = masteryLevel,
                    Progress = progress,
                };

                return (true, message, result);
            }
            catch (Exception ex)
            {
                return (false, $"Could not reach backend: {ex.Message}", null);
            }
        }

        private static (bool success, string message) PostJson(string endpoint, object payload, string successMessage)
        {
            try
            {
                string url = BuildUrl(endpoint);
                string json = JsonSerializer.Serialize(payload, JsonOptions);
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = Http.PostAsync(url, content).GetAwaiter().GetResult();
                string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                if (response.IsSuccessStatusCode)
                {
                    return (true, successMessage);
                }

                string detail = ExtractErrorMessage(body);
                return (false, string.IsNullOrWhiteSpace(detail) ? "Request failed." : detail);
            }
            catch (Exception ex)
            {
                return (false, $"Could not reach backend: {ex.Message}");
            }
        }

        private static string BuildUrl(string endpoint)
        {
            string baseUrl = BackendBaseUrl.TrimEnd('/');
            string path = endpoint.StartsWith("/") ? endpoint : $"/{endpoint}";
            return $"{baseUrl}{path}";
        }

        private static string ExtractErrorMessage(string responseBody)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(responseBody))
                {
                    return string.Empty;
                }

                using JsonDocument doc = JsonDocument.Parse(responseBody);
                if (doc.RootElement.TryGetProperty("detail", out JsonElement detail))
                {
                    return detail.GetString() ?? string.Empty;
                }

                if (doc.RootElement.TryGetProperty("message", out JsonElement message))
                {
                    return message.GetString() ?? string.Empty;
                }
            }
            catch
            {
                // fall through
            }

            return string.Empty;
        }

        private static AccountInfo ParseAccountInfo(JsonElement accountEl)
        {
            int id = accountEl.TryGetProperty("id", out JsonElement idEl) && idEl.TryGetInt32(out int parsedId)
                ? parsedId
                : 0;
            string username = accountEl.TryGetProperty("username", out JsonElement usernameEl)
                ? (usernameEl.GetString() ?? string.Empty)
                : string.Empty;
            string name = accountEl.TryGetProperty("name", out JsonElement nameEl)
                ? (nameEl.GetString() ?? string.Empty)
                : string.Empty;
            string masteryLevel = accountEl.TryGetProperty("mastery_level", out JsonElement masteryEl)
                ? (masteryEl.GetString() ?? string.Empty)
                : string.Empty;
            Dictionary<string, int> progress = accountEl.TryGetProperty("progress", out JsonElement progressEl)
                ? ParseProgress(progressEl)
                : new Dictionary<string, int>();

            return new AccountInfo
            {
                Id = id,
                Username = username,
                Name = name,
                MasteryLevel = masteryLevel,
                Progress = progress,
            };
        }

        private static Dictionary<string, int> ParseProgress(JsonElement progressEl)
        {
            var progress = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            if (progressEl.ValueKind != JsonValueKind.Object)
            {
                return progress;
            }

            foreach (JsonProperty prop in progressEl.EnumerateObject())
            {
                if (prop.Value.TryGetInt32(out int level))
                {
                    progress[prop.Name] = level;
                }
            }

            return progress;
        }
    }
}