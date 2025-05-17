# AQI Prediction Frontend

A modern web application for displaying Air Quality Index (AQI) predictions for individual users. This application provides a user-friendly interface to view AQI predictions for different cities, along with health information and recommendations.

## Features

- View AQI predictions for multiple cities
- Interactive AQI gauge visualization
- Health information based on air quality category
- City-specific imagery
- Mobile-responsive design

## Technology Stack

- **Next.js**: React framework for building the UI
- **TypeScript**: For type-safe code
- **TailwindCSS**: For styling and responsive design
- **Axios**: For API requests
- **Chart.js**: For data visualization

## Getting Started

### Prerequisites

- Node.js 14.x or higher
- npm or yarn
- Backend AQI prediction API running (on port 8000 by default)

### Installation

1. Clone the repository
2. Navigate to the project directory

```bash
cd aqi_prediction_fe
```

3. Install dependencies

```bash
npm install
# or
yarn install
```

4. Start the development server

```bash
npm run dev
# or
yarn dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

### Configuration

By default, the application proxies API requests to `http://localhost:8000`. To change this:

1. Update the `next.config.js` file:

```js
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://your-api-url/:path*',
    },
  ]
}
```

## API Integration

The frontend integrates with the AQI prediction API to fetch:

- Available scenarios (cities and pollutants)
- AQI predictions based on selected scenarios
- Health information based on AQI values

## Project Structure

- `/components`: Reusable UI components
- `/pages`: Next.js pages (routes)
- `/services`: API service functions
- `/styles`: Global CSS and styles
- `/types`: TypeScript type definitions

## Building for Production

```bash
npm run build
# or
yarn build
```

The optimized production build will be generated in the `.next` folder.

To start the production server:

```bash
npm run start
# or
yarn start
```

## License

This project is part of the AQI Prediction System. 