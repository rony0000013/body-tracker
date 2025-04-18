// @ts-check
import { defineConfig } from 'astro/config';
import solid from '@astrojs/solid-js';

import tailwindcss from '@tailwindcss/vite';

import vercel from '@astrojs/vercel';

// https://astro.build/config
export default defineConfig({
  integrations: [solid({ devtools: true })],

  vite: {
    plugins: [tailwindcss()]
  },

  adapter: vercel()
});