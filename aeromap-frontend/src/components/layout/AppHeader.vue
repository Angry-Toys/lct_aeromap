<template>
  <header class="app-header">

    <div class="logo-container">
      <div class="logo-icon-wrapper">
        <i class="fas fa-drone logo-icon"></i>
      </div>
      <h1 class="logo-text">AeroMap</h1>
    </div>

    <div class="actions">

      <button class="btn action-btn purple-btn" @click="$emit('generate-png')">
        <i class="fas fa-chart-bar icon mr-2"></i>
        Экспорт PNG
      </button>

      <button class="btn action-btn green-btn" @click="$emit('export-json')">
        <i class="fas fa-file-export icon mr-2"></i>
        Экспорт JSON
      </button>

      <button class="btn action-btn gray-btn" @click="$emit('upload-clicked')">
        <i class="fas fa-cloud-upload-alt icon mr-2"></i>
        Загрузить данные
      </button>

      <button class="logout-btn" @click="logout">
        <i class="fas fa-power-off"></i>
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router';
// Определяем, что компонент может испускать событие 'upload-clicked'
defineEmits(['upload-clicked', 'export-json', 'generate-png']);

const router = useRouter();

const logout = () => {
  localStorage.removeItem('user-token');
  // Предполагаем, что '/' — это страница входа
  router.push('/');
};
</script>

<style scoped>
/* ==================== ОБЩИЕ СТИЛИ ШАПКИ ==================== */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 64px;
  /* Убираем синий фон, который был ранее в этом файле */
  background-color: #000000;
  border-bottom: none;
  /* Добавляем легкую тень, чтобы шапка выделялась на черном фоне */
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
  z-index: 100;
}

/* ==================== 1. ЛОГОТИП ==================== */
.logo-container {
  display: flex;
  align-items: center;
  /* Градиент + размытие + рамка, как в примере Tailwind */
  background: linear-gradient(to right, rgba(60, 60, 60, 0.2), rgba(30, 30, 30, 0.2));
  border-radius: 16px; /* rounded-2xl */
  padding: 8px 16px;
  border: 1px solid rgba(120, 120, 120, 0.2);

  /* Эффект размытия фона (backdrop-blur-sm) */
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}

.logo-icon-wrapper {
  width: 32px;
  height: 32px;
  border-radius: 8px; /* rounded-lg */
  /* Градиент иконки */
  background: linear-gradient(to bottom right, #555, #333);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
}

.logo-icon {
  font-size: 0.875rem; /* text-sm */
  color: #c0c0c0; /* text-gray-200 */
}

.logo-text {
  font-size: 1.5rem; /* text-2xl */
  font-weight: 900; /* font-black */
  margin: 0;

  /* Градиентный текст (bg-clip-text text-transparent) */
  background: linear-gradient(to right, #ffffff, #e0e0e0);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-fill-color: transparent;
}

/* ==================== 2. КНОПКИ ДЕЙСТВИЙ ==================== */
.actions {
  display: flex;
  gap: 16px; /* space-x-4 */
  align-items: center;
}

.btn {
  /* Общие стили для всех кнопок */
  font-size: 1rem;
  font-weight: 700; /* font-bold */
  padding: 12px 24px;
  border-radius: 8px; /* rounded-2xl */
  transition: all 0.3s ease-in-out; /* transition-all duration-300 */
  display: flex;
  align-items: center;
  white-space: nowrap;
  cursor: pointer;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); /* shadow-lg */
  color: #ffffff;
    font-family: Inter,
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    Roboto,
    Oxygen,
    Ubuntu,
    Cantarell,
    'Fira Sans',
    'Droid Sans',
    'Helvetica Neue',
    sans-serif !important;
}

.icon {
    margin-right: 8px; /* mr-2 */
}

/* Эффект при наведении (hover:scale-105) */
.btn:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4); /* Усиление тени */
}

/* --- ЦВЕТА КНОПОК --- */

/* Фиолетовый (Generate PNG) */
.purple-btn {
  background: linear-gradient(to right, #6d28d9, #4c1d95); /* from-purple-700 to-purple-800 */
  border: 1px solid #6d28d9;
}
.purple-btn:hover {
  background: linear-gradient(to right, #7c3aed, #6d28d9); /* hover:from-purple-600 hover:to-purple-700 */
  box-shadow: 0 10px 20px rgba(109, 40, 217, 0.4); /* hover:shadow-purple-700/25 */
}

/* Зеленый (Export JSON) */
.green-btn {
  background: linear-gradient(to right, #047857, #065f46); /* from-green-700 to-green-800 */
  border: 1px solid #047857;
}
.green-btn:hover {
  background: linear-gradient(to right, #059669, #047857); /* hover:from-green-600 hover:to-green-700 */
  box-shadow: 0 10px 20px rgba(4, 120, 87, 0.4); /* hover:shadow-green-700/25 */
}

/* Серый (Upload Data) */
.gray-btn {
  background: linear-gradient(to right, #4b5563, #374151); /* from-gray-700 to-gray-800 */
  border: 1px solid #4b5563;
}
.gray-btn:hover {
  background: linear-gradient(to right, #6b7280, #4b5563); /* hover:from-gray-600 hover:to-gray-700 */
  box-shadow: 0 10px 20px rgba(75, 85, 99, 0.4); /* hover:shadow-gray-700/25 */
}

/* --- КНОПКА ВЫХОДА --- */

.logout-btn {
  /* Стиль кнопки выключения (иконка power-off) */
  padding: 12px;
  border-radius: 12px; /* rounded-xl */
  background-color: rgba(0, 0, 0, 0.5); /* bg-black/50 */
  color: #6b7280; /* text-gray-500 */
  border: 1px solid rgba(55, 65, 81, 0.5); /* border-gray-800/50 */
  transition: all 0.3s ease-in-out;
  cursor: pointer;
}

.logout-btn:hover {
  /* Эффект красного свечения при наведении */
  background-color: rgba(239, 68, 68, 0.2); /* hover:bg-red-500/20 */
  color: #f87171; /* hover:text-red-400 */
  border-color: rgba(239, 68, 68, 0.5); /* hover:border-red-500/50 */
  transform: scale(1.1); /* Легкое увеличение */
}

/* Эффект "прыгающей" иконки при наведении на кнопки действий */
.action-btn:hover .icon {
    /* Имитация group-hover:animate-bounce */
    animation: bounce-once 0.6s;
}

@keyframes bounce-once {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-5px);
    }
}
</style>
