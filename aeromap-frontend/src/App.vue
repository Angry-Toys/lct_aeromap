<template>
  <div id="app">
    <h1>Анализ полетов БПЛА</h1>
    <MapOverview :flightData="sampleData" />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import MapOverview from '@/components/MapOverview.vue';

const sampleData = ref([
  { name: 'Москва', value: 500 },
  { name: 'Санкт-Петербург', value: 300 },
]);
</script>

<style>
#app {
  font-family: Arial, sans-serif;
  text-align: center;
  padding: 20px;
  background: #f0f0f0;
}
h1 {
  color: #2d8aff;
}
</style>
