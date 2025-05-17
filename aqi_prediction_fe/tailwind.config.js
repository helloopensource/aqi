/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'aqi-good': '#00E400',
        'aqi-moderate': '#FFFF00',
        'aqi-sensitive': '#FF7E00',
        'aqi-unhealthy': '#FF0000',
        'aqi-very-unhealthy': '#8F3F97',
        'aqi-hazardous': '#7E0023',
      },
    },
  },
  plugins: [],
} 