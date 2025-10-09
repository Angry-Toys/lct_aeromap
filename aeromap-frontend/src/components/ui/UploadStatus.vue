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
import type { PropType } from 'vue'; // Тип
import { watch } from 'vue'; // Значение

// Тип для задачи загрузки
interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'success' | 'error' | 'timeout';
}

const props = defineProps({
  tasks: {
    type: Array as PropType<UploadTask[]>,
    required: true
  }
});

// Функция для текста статуса
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

// Автоудаление задач через 10 секунд после завершения
watch(
  () => props.tasks,
  (newTasks) => {
    newTasks.forEach((task) => {
      if (task.status === 'success' || task.status === 'error' || task.status === 'timeout') {
        setTimeout(() => {
          const taskIndex = props.tasks.findIndex(t => t.id === task.id);
          if (taskIndex !== -1) {
            props.tasks.splice(taskIndex, 1);
          }
        }, 10000); // 10 секунд
      }
    });
  },
  { deep: true }
);
</script>

<style scoped>
.upload-status-container {
  position: fixed;
  bottom: 20px;
  right: 40px;
  width: 350px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  z-index: 2000;
}

.upload-task {
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(17, 17, 17, 0.9));
  border: 1px solid #333333;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
  transition: all 0.3s ease;
}

.upload-task:hover {
  border-color: #ffc107;
  box-shadow: 0 6px 20px rgba(255, 193, 7, 0.3);
}

.task-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: #ffffff;
}

.filename {
  font-size: 0.95rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
  color: #e0e0e0;
}

.status {
  font-size: 0.9rem;
  font-weight: 600;
}

.status.uploading { color: #ffc107; }
.status.success { color: #30e2aa; }
.status.error, .status.timeout { color: #ff6666; }

.progress-bar-wrapper {
  height: 6px;
  background: #1a1a1a;
  border-radius: 3px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  transition: width 0.3s ease;
  background: linear-gradient(90deg, #ffac30, #ffc107);
}

.progress-bar.success {
  background: linear-gradient(90deg, #30e2aa, #059669);
}

.progress-bar.error, .progress-bar.timeout {
  background: linear-gradient(90deg, #ff6666, #b53b3b);
}
</style>
