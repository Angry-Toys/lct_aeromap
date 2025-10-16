<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-content">
      <h3 class="modal-title">Загрузка нового отчёта</h3>
      <p class="modal-subtitle">Поддерживаются только файлы формата .xlsx</p>

      <div
        class="drop-zone"
        :class="{ 'is-active': isDragActive }"
        @dragover.prevent="isDragActive = true"
        @dragleave.prevent="isDragActive = false"
        @drop.prevent="handleDrop"
        @click="openFileBrowser"
      >
        <input type="file" ref="fileInput" @change="handleFileSelect" accept=".xlsx" hidden />

        <div class="drop-zone-content">
          <svg class="upload-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M11 15V11H7V9H11V5H13V9H17V11H13V15H11ZM12 22C10.8167 22 9.70833 21.7833 8.675 21.35C7.64167 20.9167 6.75 20.25 6 19.35C5.25 18.45 4.68333 17.4333 4.3 16.3C3.91667 15.1667 3.725 13.9667 3.725 12.7C3.725 11.2333 4.08333 9.9 4.8 8.7C5.51667 7.5 6.45833 6.55 7.625 5.85C8.79167 5.15 10.0583 4.8 11.425 4.8C12.5583 4.8 13.625 5.01667 14.625 5.45C15.625 5.88333 16.5 6.46667 17.25 7.2L15.825 8.625C15.325 8.125 14.7667 7.75 14.15 7.5C13.5333 7.25 12.8833 7.125 12.2 7.125C10.8667 7.125 9.725 7.575 8.775 8.475C7.825 9.375 7.35 10.4833 7.35 11.8C7.35 12.1333 7.3625 12.4583 7.3875 12.775C7.4125 13.0917 7.45 13.35 7.5 13.55L9.375 11.675L10.8 13.1L7.875 16.025C8.025 16.175 8.1875 16.3125 8.3625 16.4375C8.5375 16.5625 8.725 16.675 8.925 16.775L11.75 13.95L13.175 15.375L10.25 18.3C11.1833 18.9667 12.2333 19.3 13.4 19.3C14.7333 19.3 15.875 18.85 16.825 17.95C17.775 17.05 18.25 15.9333 18.25 14.6C18.9167 14.5333 19.5083 14.2875 20.025 13.8625C20.5417 13.4375 20.8 12.9 20.8 12.25C20.8 11.55 20.55 10.9583 20.05 10.475C19.55 9.99167 18.95 9.75 18.25 9.75H17.125C16.925 8.98333 16.5833 8.3 16.1 7.7C15.6166 7.1 15.025 6.61667 14.325 6.25L15.35 5.225C16.2167 5.75833 16.975 6.45 17.625 7.3C18.275 8.15 18.6 9.08333 18.6 10.1L18.725 10.225C19.1917 10.2917 19.575 10.5 19.875 10.85C20.175 11.2 20.325 11.6 20.325 12.05C20.325 12.75 20.0583 13.3 19.525 13.7C18.9917 14.1 18.35 14.3 17.6 14.3C17.0667 15.6333 16.1917 16.675 14.975 17.425C13.7583 18.175 12.4333 18.55 11 18.55C9.9 18.55 8.86667 18.325 7.9 17.875L9.325 19.3C9.99167 19.8333 10.7583 20.1875 11.625 20.3625C12.4917 20.5375 13.3583 20.45 14.225 20.1L12 22.325C10.0333 22.325 8.28333 21.7 6.75 20.45C5.21667 19.2 4.25 17.5833 3.85 15.6C3.45 13.6167 3.63333 11.7 4.4 9.85C5.16667 7.96667 6.41667 6.45 8.15 5.3C9.88333 4.15 11.8333 3.575 14 3.575C14.7 3.575 15.3667 3.65833 16 3.825L17.275 2.55C16.075 1.91667 14.775 1.55 13.375 1.45C11.975 1.35 10.5583 1.55 9.125 2.05C7.69167 2.55 6.40833 3.3 5.275 4.3C4.14167 5.3 3.25 6.5 2.6 7.9C1.95 9.3 1.625 10.8 1.625 12.4C1.625 13.9333 1.88333 15.35 2.4 16.65C2.91667 17.95 3.65 19.0667 4.6 19.9917C5.55 20.9167 6.66667 21.5833 7.95 21.9917C9.23333 22.4 10.5667 22.6 12 22.6L12 22Z" /></svg>
          <p v-if="!selectedFile">Перетащите XLSX файл сюда или <span class="browse-link">выберите файл</span></p>
          <div v-else class="file-info">
            <p><span class="file-name">{{ selectedFile.name }}</span></p>
            <p class="file-size">{{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB</p>
          </div>
        </div>
      </div>

      <div class="modal-actions">
        <button class="btn secondary" @click="$emit('close')">Отмена</button>
        <button class="btn primary" @click="startUpload" :disabled="!selectedFile">
          Начать загрузку
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const emit = defineEmits(['start-upload', 'close']);
const fileInput = ref<HTMLInputElement | null>(null);
const selectedFile = ref<File | null>(null);
const isDragActive = ref(false);

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    selectedFile.value = target.files[0];
  }
};

const handleDrop = (event: DragEvent) => {
  isDragActive.value = false;
  if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
    selectedFile.value = event.dataTransfer.files[0];
  }
};

const openFileBrowser = () => {
  if (!selectedFile.value) {
    fileInput.value?.click();
  }
};

const startUpload = () => {
  if (selectedFile.value) {
    emit('start-upload', selectedFile.value);
    emit('close');
  }
};
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}
.modal-content {
  background: rgba(17, 17, 17, 0.8);
  padding: 32px;
  border-radius: 16px;
  border: 1px solid rgba(50, 50, 50, 0.5);
  width: 90%;
  max-width: 500px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  overflow-y: auto;
  animation: fadeInScale 0.1s ease-out forwards;
}
.modal-title {
  margin: 0 0 8px;
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
  text-align: center;
}
.modal-subtitle {
  margin: 0 0 24px;
  text-align: center;
  color: #aaaaaa;
  font-size: 0.9rem;
}
.drop-zone {
  border: 2px dashed #444;
  padding: 30px;
  text-align: center;
  border-radius: 12px;
  margin-bottom: 24px;
  color: #aaaaaa;
  transition: all 0.3s ease;
  cursor: pointer;
  background-color: rgba(0,0,0,0.2);
}
.drop-zone:hover, .drop-zone.is-active {
  border-color: #ffc107;
  background-color: rgba(255, 193, 7, 0.05);
}
.drop-zone-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}
.upload-icon {
  width: 48px;
  height: 48px;
  color: #666;
  transition: color 0.3s ease;
}
.drop-zone:hover .upload-icon, .drop-zone.is-active .upload-icon {
  color: #ffc107;
}
.browse-link {
  color: #ffc107;
  font-weight: 500;
  text-decoration: none;
}
.file-info {
  color: #e0e0e0;
  font-size: 1rem;
}
.file-name {
  font-weight: 600;
  color: #ffc107;
}
.file-size {
  font-size: 0.8rem;
  color: #aaaaaa;
}
.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}
.btn {
  padding: 12px 24px;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
}
.btn.primary {
  color: #000;
  background-color: #ffc107;
}
.btn.primary:hover:not(:disabled) {
  background-color: #ffac30;
  box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
}
.btn.primary:disabled {
  background-color: #333;
  color: #666;
  cursor: not-allowed;
}
.btn.secondary {
  color: #e0e0e0;
  background-color: #2a2a2a;
}
.btn.secondary:hover {
  background-color: #333;
}

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
</style>
