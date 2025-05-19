import asyncio
import base64
from fastmcp import Client

async def main():
    # Connect to the server using SSE transport (default)
    client_url = "http://localhost:8000/sse"
    
    try:
        # Create client with default transport
        async with Client(client_url) as client:
            # List available tools
            print("Tools:", await client.list_tools()) 

            print("\nCalling learn_and_plot_causal_graph tool...")
            result = await client.call_tool(
                "learn_and_plot_causal_graph",
                {
                    "csv_path": "Lung_Cancer.csv",
                    "selected_cols": [
                        "AGE", "SEX", "EXDOSE", "EXDOSU",
                        "EXROUTE", "HEIGHT", "WEIGHT"
                    ]
                }
            )

            print(f"\nResult type: {type(result)}")
            
            # Handle dictionary with base64-encoded image
            if isinstance(result, dict) and 'image_base64' in result:
                try:
                    # Decode the base64 string
                    img_data = base64.b64decode(result['image_base64'])
                    
                    # Save to file
                    output_path = "causal_graph.png"
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                        
                    print(f"\nSuccess! Causal graph saved to: {output_path}")
                    print(f"Image size: {len(img_data)} bytes")
                    
                except Exception as e:
                    print(f"\nError decoding base64 image: {e}")
            else:
                print("\nUnexpected result format from server:")
                print(f"Result: {result}")
    except Exception as e:
        print(f"\nError connecting to the server: {e}")

if __name__ == "__main__":
    asyncio.run(main())