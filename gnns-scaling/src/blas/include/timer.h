#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

/*
Timer timer;
timer.start();
int counter = 0;
double test, test2;
while(timer.elapsedSeconds() < 10.0)
{
    counter++;
    test = std::cos(counter / M_PI);
    test2 = std::sin(counter / M_PI);
}
timer.stop();

std::cout << counter << std::endl;
std::cout << "Seconds: " << timer.elapsedSeconds() << std::endl;
std::cout << "Milliseconds: " << timer.elapsedMilliseconds() << std::endl ;
*/

class Timer
{
public:
    void start()
    {
        m_StartTime = std::chrono::system_clock::now();
        m_bRunning = true;
    }
    
    void stop()
    {
        m_EndTime = std::chrono::system_clock::now();
        m_bRunning = false;
    }
    
    double elapsedMilliseconds()
    {
        std::chrono::time_point<std::chrono::system_clock> endTime;
        
        if(m_bRunning)
        {
            endTime = std::chrono::system_clock::now();
        }
        else
        {
            endTime = m_EndTime;
        }
        
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
    }
    
    double elapsedSeconds()
    {
        return elapsedMilliseconds() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool                                               m_bRunning = false;
};
