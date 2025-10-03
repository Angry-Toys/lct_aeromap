<template>
  <header class="app-header">
    <h1 class="logo">DroneWatch MVP</h1>
    <div class="actions">
      <button class="btn primary" @click="$emit('upload-clicked')">Загрузить отчёт</button>
      <button class="btn secondary" @click="logout">Выйти</button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router';
defineEmits(['upload-clicked']);
const router = useRouter();
const logout = () => {
  localStorage.removeItem('user-token');
  router.push('/');
};
</script>

<style scoped>
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background-color: #0f2346;
  border-bottom: 1px solid #226bcb;
}
.logo {
  font-size: 1.5rem;
  font-weight: 600;
  color: #e0e0e0;
}
.actions {
  display: flex;
  gap: 16px;
}
.btn {
  padding: 10px 20px;
  font-size: 0.9rem;
  font-weight: 500;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.btn.primary {
  color: #fff;
  background-color: #30ceda;
}
.btn.primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(48, 206, 218, 0.4);
}
.btn.secondary {
  color: #a0c3ff;
  background-color: transparent;
  border: 1px solid #226bcb;
}
.btn.secondary:hover {
  background-color: #226bcb;
  color: #fff;
}
</style>
