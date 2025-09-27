import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { MapChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, VisualMapComponent, ToolboxComponent } from 'echarts/components';
echarts.use([CanvasRenderer, MapChart, TitleComponent, TooltipComponent, VisualMapComponent, ToolboxComponent]);

const app = createApp(App)

app.config.globalProperties.$echarts = echarts;  // Глобальный доступ

app.use(createPinia())
app.use(router)

app.mount('#app')
