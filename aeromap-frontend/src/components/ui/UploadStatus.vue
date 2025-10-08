<template>
  <div class="upload-status-container" v-if="tasks.length > 0">
    <div v-for="task in tasks" :key="task.id" class="upload-task">
      <div class="task-info">
        <span class="filename">{{ task.file.name }}</span>
        <span class="status" :class="task.status">{{ getStatusText(task.status, task.progress) }}</span>
      </div>
      <div class="progress-bar-wrapper">
        <div
          class="progress-bar"
          :class="task.status"
          :style="{ width: task.progress + '%' }"
        ></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { PropType } from 'vue';

// Определяем тип для задачи загрузки
interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'success' | 'error' | 'timeout';
}

defineProps({
  tasks: {
    type: Array as PropType<UploadTask[]>,
    required: true
  }
});

// Функция для отображения текста статуса
const getStatusText = (status: UploadTask['status'], progress: number) => {
  switch (status) {
    case 'uploading':
      return `Загрузка... ${progress}%`;
    case 'success':
      return '✓ Успешно';
    case 'error':
      return '✗ Ошибка';
    case 'timeout':
      return '✗ Время ожидания истекло';
    default:
      return '';
  }
};
</script>

<style scoped>
.upload-status-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 350px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  z-index: 2000;
}
.upload-task {
  /* Фон: Темно-серый */
  background-color: #111111;
  /* Рамка: Темно-серая */
  border: 1px solid #333333;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
}
.task-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: #fff;
}
.filename {
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
}
.status {
  font-size: 0.9rem;
  font-weight: 600;
}
/* Цвета статусов */
.status.uploading { color: #ffc107; } /* Янтарный */
.status.success { color: #30e2aa; }  /* Зеленый */
.status.error, .status.timeout { color: #ff6666; } /* Красный */

.progress-bar-wrapper {
  height: 6px;
  background-color: #333333;
  border-radius: 3px;
  overflow: hidden;
}
.progress-bar {
  height: 100%;
  transition: width 0.3s ease;
}
/* Цвета прогресса */
.progress-bar.uploading { background-color: #ffc107; }
.progress-bar.success { background-color: #30e2aa; }
.progress-bar.error, .progress-bar.timeout { background-color: #ff6666; }
</style>
