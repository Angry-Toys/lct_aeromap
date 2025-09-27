import type {
  ComposeOption,
  TooltipComponentOption,
  VisualMapComponentOption,
  ToolboxComponentOption,
  GeoComponentOption,
} from "echarts"
import type { MapSeriesOption, ScatterSeriesOption } from "echarts/charts"

// Общий тип для опции графика
export type ECOption = ComposeOption<
  | TooltipComponentOption
  | VisualMapComponentOption
  | ToolboxComponentOption
  | GeoComponentOption
  | MapSeriesOption
  | ScatterSeriesOption
>
