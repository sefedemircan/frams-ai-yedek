import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

const api = {
  // Depoları getir
  getDepots: async () => {
    try {
      const response = await axios.get(`${API_URL}/depots`);
      return response.data;
    } catch (error) {
      console.error('Depolar alınırken hata oluştu:', error);
      throw error;
    }
  },

  // Araçları getir
  getVehicles: async () => {
    try {
      const response = await axios.get(`${API_URL}/vehicles`);
      return response.data;
    } catch (error) {
      console.error('Araçlar alınırken hata oluştu:', error);
      throw error;
    }
  },

  // Rota optimizasyonu yap
  optimizeRoutes: async (emergencies, considerTraffic = true) => {
    try {
      const response = await axios.post(`${API_URL}/optimize`, { 
        emergencies,
        considerTraffic 
      });
      return response.data;
    } catch (error) {
      console.error('Rota optimizasyonu yapılırken hata oluştu:', error);
      throw error;
    }
  }
};

export default api; 