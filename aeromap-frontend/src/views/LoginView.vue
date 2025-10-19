<template>
  <div class="login-page">
    <div class="login-card">

      <!-- 1. Header Section -->
      <div class="header-section">
        <div class="icon-wrapper">
          <!-- Drone Icon SVG -->

        </div>
        <h1 class="title">AeroMap</h1>
        <p class="subtitle">Advanced Flight Analytics</p>
      </div>

      <!-- 2. Form Section -->
      <div class="form-section">
        <form @submit.prevent="handleLogin">

          <!-- Username Input -->
          <div class="input-group">
            <div class="input-glow"></div>
            <input
              id="username"
              type="text"
              class="input-field peer"
              placeholder=""
              v-model="username"
              required
            />
            <label for="username" class="input-label">Username</label>
            <div class="input-icon-wrapper">
              <!-- User Icon SVG -->
              <svg xmlns="http://www.w3.org/2000/svg" class="input-icon" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
              </svg>
            </div>
          </div>

          <!-- Password Input -->
          <div class="input-group">
            <div class="input-glow"></div>
            <input
              id="password"
              type="password"
              class="input-field peer"
              placeholder=""
              v-model="password"
              required
            />
            <label for="password" class="input-label">Password</label>
            <div class="input-icon-wrapper">
              <!-- Lock Icon SVG -->
              <svg xmlns="http://www.w3.org/2000/svg" class="input-icon" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clip-rule="evenodd" />
              </svg>
            </div>
          </div>

          <!-- Submit Button -->
          <button type="submit" class="submit-button group">
            <span class="shine-effect"></span>

            Запустить дашборд
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';

const username = ref('');
const password = ref('');
const router = useRouter();

const handleLogin = () => {
  // Mock successful login
  console.log('Attempting login with:', username.value);
  localStorage.setItem('user-token', 'mock-token');
  router.push('/dashboard');
};
</script>

<style scoped>
/* --- Main Page Layout --- */
.login-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  position: relative;
  overflow: hidden; /* Prevents page scrolling */
  background: linear-gradient(135deg, rgb(0, 0, 0) 0%, rgb(10, 10, 10) 50%, rgb(26, 26, 26) 100%);
}

.login-card {
  width: 100%;
  max-width: 450px;
  position: relative;
  z-index: 10;
  padding: 0 16px; /* Prevents touching edges on mobile */
}

/* --- Header Section (Icon, Title, Subtitle) --- */
.header-section {
  text-align: center;
  margin-bottom: 3rem; /* 48px */
}

.icon-wrapper {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 80px;
  height: 80px;
  border-radius: 1.5rem; /* 24px */
  background: linear-gradient(to bottom right, #333, #000);
  margin-bottom: 1.5rem; /* 24px */
  box-shadow: 0 25px 50px -12px rgba(100, 116, 139, 0.25);
  border: 1px solid #1f2937;
}

.drone-icon {
  width: 36px;
  height: 36px;
  color: #d1d5db; /* gray-300 */
}

.title {
  font-size: 3rem; /* 48px */
  font-weight: 900;
  background: linear-gradient(to right, #e5e7eb, #f9fafb, #fff);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin: 0 0 0.75rem 0; /* 12px */
}

.subtitle {
  font-size: 1.125rem; /* 18px */
  font-weight: 500;
  color: #6b7280; /* gray-500 */
  margin: 0;
}

/* --- Form Section (Glassmorphism) --- */
.form-section {
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  background-color: rgba(0, 0, 0, 0.6); /* Charcoal gray transparent background */
  border-radius: 1.5rem; /* 24px */
  padding: 2rem; /* 32px */
  border: 1px solid rgba(17, 24, 39, 0.5);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.45); /* Darker blur shadow */
  position: relative;
}

.form-section form {
  display: flex;
  flex-direction: column;
  gap: 2rem; /* 32px */
}

/* --- Input Field Styling --- */
.input-group {
  position: relative;
}

.input-glow {
  position: absolute;
  inset: 0;
  background: linear-gradient(to right, rgba(107, 114, 128, 0.1), rgba(75, 85, 99, 0.1));
  border-radius: 1rem; /* 16px */
  filter: blur(12px);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.input-group:hover .input-glow {
  opacity: 1;
}

.input-field {
  box-sizing: border-box; /* Prevents overflow */
  position: relative;
  width: 100%;
  font-size: 1rem;
  padding: 1rem 3rem 1rem 1.5rem; /* 16px 48px 16px 24px */
  background-color: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(55, 65, 81, 0.5);
  border-radius: 1rem; /* 16px */
  color: #fff;
  transition: all 0.3s ease;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}

/* --- NEW: Keyframe for focus flash animation --- */
@keyframes flash-border {
  0% {
    box-shadow: 0 0 0 4px rgba(75, 85, 99, 0.1);
    border-color: rgba(75, 85, 99, 0.5);
  }
  25% {
    border-color: rgba(255, 255, 255, 0.9);
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.9), 0 0 0 4px rgba(75, 85, 99, 0.1);
  }
  100% {
    box-shadow: 0 0 0 4px rgba(75, 85, 99, 0.1);
    border-color: rgba(75, 85, 99, 0.5);
  }
}

.input-field:focus {
  outline: none;
  border-color: rgba(75, 85, 99, 0.5);
  box-shadow: 0 0 0 4px rgba(75, 85, 99, 0.1); /* Equivalent to ring */
  animation: flash-border 0.4s ease-out; /* Apply animation */
}

/* --- Floating Label Logic --- */
.input-label {
  position: absolute;
  left: 1.5rem; /* 24px */
  top: 0.85rem; /* 16px */
  color: #6b7280; /* gray-600 */
  font-weight: 500;
  pointer-events: none;
  transition: all 0.3s ease;
  background-color: transparent;
  padding: 0;
}

.input-field:focus + .input-label,
.input-field:not(:placeholder-shown) + .input-label {
  top: -0.75rem; /* -12px */
  left: 1.25rem; /* 20px */
  font-size: 1rem; /* 14px */
  color: #9ca3af; /* gray-400 */
  background-color: #000000;
  padding: 0 0.5rem; /* 8px */
}

/* --- Input Icon --- */
.input-icon-wrapper {
  position: absolute;
  right: 1rem; /* 16px */
  top: 50%;
  transform: translateY(-50%);
  color: #6b7280;
  pointer-events: none;
  transition: color 0.3s ease;
}

.input-icon {
  width: 20px;
  height: 20px;
}

.input-field:focus ~ .input-icon-wrapper {
  color: #d1d5db;
}

/* --- Submit Button Styling --- */
.submit-button {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  background: linear-gradient(to right, #374151, #1f2937); /* gray-700 to gray-800 */
  color: #fff;
  font-weight: 700;
  font-size: 1rem;
  padding: 1rem 1.5rem;
  border-radius: 1rem;
  border: 1px solid #374151;
  transition: all 0.3s ease;
  transform-origin: center;
  cursor: pointer;
  overflow: hidden; /* Crucial for the shine effect */
}

.submit-button:hover {
  transform: scale(1.02);
  box-shadow: 0 25px 50px -12px rgba(75, 85, 99, 0.25);
  background: linear-gradient(to right, #4b5563, #374151); /* hover state */
}

.button-icon {
  width: 20px;
  height: 20px;
  margin-right: 0.75rem;
}

/* --- Shine Animation on Button Hover --- */
.shine-effect {
  position: absolute;
  inset: 0;
  background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.1), transparent);
  transform: translateX(-100%) skewX(-25deg);
  transition: transform 1s ease;
}

.submit-button:hover .shine-effect {
  transform: translateX(100%) skewX(-25deg);
}

</style>

