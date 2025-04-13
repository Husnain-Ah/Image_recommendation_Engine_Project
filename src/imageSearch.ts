export async function searchLocalImages(keyword: string): Promise<string[]> {
    const serverUrl = "http://localhost:3000";
  
    try {
      const response = await fetch(`${serverUrl}/search-images`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ keyword }),
      });
  
      if (!response.ok) {
        console.error(`Server error: ${response.status}`);
        return [];
      }
  
      const { results } = await response.json();
      return results;
    } catch (error) {
      console.error("Error searching local images:", error);
      return [];
    }
  }