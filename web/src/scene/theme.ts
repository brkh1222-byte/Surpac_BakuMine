import type { Theme } from '../state/store'

/** Colours used inside the WebGL canvas. The DOM chrome is themed in index.css. */
export interface SceneColors {
  background: string
  gridMinor: string
  gridMajor: string
  gridBorder: string
}

export const SCENE_COLORS: Record<Theme, SceneColors> = {
  dark: {
    background: '#15181c',
    gridMinor: '#262b31',
    gridMajor: '#39414a',
    gridBorder: '#77838f',
  },
  light: {
    background: '#f4f5f3',
    gridMinor: '#e1e4e0',
    gridMajor: '#c4c9c4',
    gridBorder: '#646e76',
  },
}
