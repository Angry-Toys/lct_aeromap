<template>
  <div class="login-container">
    <div class="login-box">
      <h1 class="title">DroneWatch</h1>
      <p class="subtitle">Аналитическая панель</p>
      <form @submit.prevent="handleLogin">
        <input type="text" placeholder="Логин" v-model="username" required />
        <input type="password" placeholder="Пароль" v-model="password" required />
        <button type="submit">Войти</button>
      </form>
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
  // Имитация успешной авторизации
  console.log('Попытка входа с:', username.value);
  // В реальном приложении здесь будет запрос к API
  localStorage.setItem('user-token', 'mock-token'); // Сохраняем "токен"
  router.push('/dashboard');
};
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #0a1929;
}
.login-box {
  background: #0f2346;
  padding: 40px;
  border-radius: 12px;
  border: 1px solid #226bcb;
  text-align: center;
  width: 350px;
}
.title { font-size: 2rem; color: #fff; }
.subtitle { color: #a0c3ff; margin-bottom: 30px; }
input {
  width: 100%;
  padding: 12px;
  margin-bottom: 15px;
  background: #0a1929;
  border: 1px solid #226bcb;
  border-radius: 8px;
  color: #fff;
}
button {
  width: 100%;
  padding: 12px;
  background-color: #30ceda;
  border: none;
  border-radius: 8px;
  color: #fff;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.2s;
}
button:hover { background-color: #2ab8c5; }
</style>
