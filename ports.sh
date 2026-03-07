#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking for processes on ports 5000, 5001, 5002...${NC}"

# Array of ports to check
ports=(5000 5001 5002)

for port in "${ports[@]}"; do
    echo -e "\n${YELLOW}Port $port:${NC}"
    
    # Find PID using the port (works on both macOS and Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        pid=$(lsof -ti :$port)
    else
        # Linux
        pid=$(ss -lptn 'sport = :$port' | grep -oP '(?<=pid=)\d+' 2>/dev/null || lsof -ti :$port 2>/dev/null)
    fi
    
    if [ -z "$pid" ]; then
        echo -e "${GREEN}  ✓ No process found on port $port${NC}"
    else
        echo -e "  Found process(es) with PID(s): $pid"
        
        # Kill each PID
        for p in $pid; do
            process_name=$(ps -p $p -o comm= 2>/dev/null)
            echo -e "  ${YELLOW}Killing process $p ($process_name) on port $port${NC}"
            
            # Try graceful kill first, then force kill
            kill $p 2>/dev/null
            sleep 1
            
            # Check if process still exists and force kill if needed
            if kill -0 $p 2>/dev/null; then
                echo -e "  ${RED}Process $p still running, using force kill...${NC}"
                kill -9 $p 2>/dev/null
            fi
        done
        
        # Verify it's killed
        sleep 1
        if [[ "$OSTYPE" == "darwin"* ]]; then
            remaining=$(lsof -ti :$port)
        else
            remaining=$(lsof -ti :$port 2>/dev/null)
        fi
        
        if [ -z "$remaining" ]; then
            echo -e "  ${GREEN}✓ Successfully killed all processes on port $port${NC}"
        else
            echo -e "  ${RED}✗ Failed to kill some processes on port $port${NC}"
        fi
    fi
done

echo -e "\n${GREEN}Done!${NC}"