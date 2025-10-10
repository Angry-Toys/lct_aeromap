<template>
  <div class="upload-status-container">
    <transition-group name="list">
      <div v-for="task in tasks" :key="task.id" class="upload-task">
        <div class="task-icon">
          <!-- Success -->
          <svg v-if="task.status === 'success'" class="icon-success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" /></svg>
          <!-- Error / Timeout -->
          <svg v-else-if="task.status === 'error' || task.status === 'timeout'" class="icon-error" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>
          <!-- Processing Spinner -->
          <div v-else-if="task.status === 'processing'" class="spinner"></div>
          <!-- Uploading -->
          <svg v-else class="icon-uploading" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l-3.75 3.75M12 9.75l3.75 3.75M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
        </div>
        <div class="task-details">
          <div class="task-info">
            <span class="filename">{{ task.file.name }}</span>
            <span class="status" :class="task.status">{{ getStatusText(task.status) }}</span>
          </div>
          <!-- MODIFIED: Added a block to display the error message -->
          <div v-if="(task.status === 'error' || task.status === 'timeout') && task.errorMessage" class="error-message">
            {{ task.errorMessage }}
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
    </transition-group>
  </div>
</template>

<script setup lang="ts">
import type { PropType } from 'vue';
import { watch } from 'vue';

interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'success' | 'error' | 'timeout';
  errorMessage?: string; // <-- MODIFIED: Added errorMessage field
}

const props = defineProps({
  tasks: {
    type: Array as PropType<UploadTask[]>,
    required: true
  }
});

// MODIFIED: getStatusText no longer needs progress for the 'uploading' status text
const getStatusText = (status: UploadTask['status']) => {
  switch (status) {
    case 'uploading':
      return `Загрузка...`;
    case 'processing':
      return 'Обработка...';
    case 'success':
      return 'Успешно';
    case 'error':
      return 'Ошибка';
    case 'timeout':
      return 'Время истекло';
    default:
      return '';
  }
};

watch(() => props.tasks, (newTasks) => {
  newTasks.forEach((task) => {
    if (['success', 'error', 'timeout'].includes(task.status)) {
      setTimeout(() => {
        const taskIndex = props.tasks.findIndex(t => t.id === task.id);
        if (taskIndex !== -1) {
          props.tasks.splice(taskIndex, 1);
        }
      }, 5000);
    }
  });
}, { deep: true });
</script>

<style scoped>
.upload-status-container {
  position: fixed;
  bottom: 20px;
  right: 80px;
  width: 350px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  z-index: 2000;
}

.upload-task {
  display: flex;
  align-items: center;
  gap: 16px;
  background: linear-gradient(135deg, rgba(17, 17, 17, 0.8), rgba(0, 0, 0, 0.9));
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
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

.task-icon {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-success { color: #30e2aa; }
.icon-error { color: #ff6666; }
.icon-uploading { color: #ffc107; }

.task-details {
  flex-grow: 1;
  min-width: 0;
}

.task-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

/* MODIFIED: Added styles for the error message text */
.error-message {
  font-size: 0.8rem;
  color: #ff9a9a;
  margin-top: -8px;
  margin-bottom: 8px;
  white-space: normal;
  word-break: break-word;
}

.filename {
  font-size: 0.95rem;
  font-weight: 500;
  color: #e0e0e0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 16px;
}

.status {
  font-size: 0.9rem;
  font-weight: 600;
  white-space: nowrap;
}

.status.uploading, .status.processing { color: #ffc107; }
.status.success { color: #30e2aa; }
.status.error, .status.timeout { color: #ff6666; }

.progress-bar-wrapper {
  height: 6px;
  background: #2a2a2a;
  border-radius: 3px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-bar.uploading { background: #ffc107; }
.progress-bar.success { background: #30e2aa; }
.progress-bar.error, .progress-bar.timeout { background: #ff6666; }
.progress-bar.processing {
  background: repeating-linear-gradient(
    45deg,
    #ffc107,
    #ffc107 10px,
    #ffac30 10px,
    #ffac30 20px
  );
  background-size: 200% 200%;
  animation: progress-indeterminate 1s linear infinite;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #ffac30;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes progress-indeterminate {
  0% { background-position: 0% 50%; }
  100% { background-position: -100% 50%; }
}

.list-enter-active, .list-leave-active {
  transition: all 0.5s cubic-bezier(0.55, 0, 0.1, 1);
}
.list-enter-from, .list-leave-to {
  opacity: 0;
  transform: scale(0.8) translateX(30px);
}
</style>

