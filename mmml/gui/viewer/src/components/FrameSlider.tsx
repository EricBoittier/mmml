import { useState, useEffect, useRef, useCallback } from 'react';

interface FrameSliderProps {
  currentFrame: number;
  totalFrames: number;
  onFrameChange: (frame: number) => void;
}

function FrameSlider({ currentFrame, totalFrames, onFrameChange }: FrameSliderProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(10); // frames per second
  const intervalRef = useRef<number | null>(null);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    onFrameChange(value);
  };

  const handlePrevFrame = () => {
    if (currentFrame > 0) {
      onFrameChange(currentFrame - 1);
    }
  };

  const handleNextFrame = () => {
    if (currentFrame < totalFrames - 1) {
      onFrameChange(currentFrame + 1);
    }
  };

  const handleFirstFrame = () => {
    onFrameChange(0);
  };

  const handleLastFrame = () => {
    onFrameChange(totalFrames - 1);
  };

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // Use ref to track current frame for animation to avoid stale closures
  const frameRef = useRef(currentFrame);
  useEffect(() => {
    frameRef.current = currentFrame;
  }, [currentFrame]);

  // Animation loop
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(() => {
        const next = frameRef.current + 1;
        if (next >= totalFrames) {
          setIsPlaying(false);
          onFrameChange(0); // Loop back to start
        } else {
          onFrameChange(next);
        }
      }, 1000 / playSpeed);
    } else if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, playSpeed, totalFrames, onFrameChange]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      
      switch (e.key) {
        case 'ArrowLeft':
          handlePrevFrame();
          break;
        case 'ArrowRight':
          handleNextFrame();
          break;
        case ' ':
          e.preventDefault();
          togglePlay();
          break;
        case 'Home':
          handleFirstFrame();
          break;
        case 'End':
          handleLastFrame();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentFrame, totalFrames, togglePlay]);

  return (
    <div className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 px-4 py-3">
      <div className="flex items-center gap-4">
        {/* Playback controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleFirstFrame}
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400"
            title="First frame (Home)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
          
          <button
            onClick={handlePrevFrame}
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400"
            title="Previous frame (Left arrow)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          
          <button
            onClick={togglePlay}
            className={`p-2 rounded ${isPlaying ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400' : 'hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400'}`}
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          >
            {isPlaying ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </button>
          
          <button
            onClick={handleNextFrame}
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400"
            title="Next frame (Right arrow)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
          
          <button
            onClick={handleLastFrame}
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400"
            title="Last frame (End)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        {/* Frame slider */}
        <div className="flex-1 flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={currentFrame}
            onChange={handleSliderChange}
            className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>

        {/* Frame counter */}
        <div className="text-sm text-slate-600 dark:text-slate-400 min-w-[100px] text-right font-mono">
          {currentFrame + 1} / {totalFrames}
        </div>

        {/* Speed control */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 dark:text-slate-400">Speed:</label>
          <select
            value={playSpeed}
            onChange={(e) => setPlaySpeed(parseInt(e.target.value, 10))}
            className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
          >
            <option value={1}>1 fps</option>
            <option value={5}>5 fps</option>
            <option value={10}>10 fps</option>
            <option value={20}>20 fps</option>
            <option value={30}>30 fps</option>
          </select>
        </div>
      </div>
    </div>
  );
}

export default FrameSlider;
