-- Create tables for warehouse stock counting system
-- Run this script to set up the database schema

-- Table for storing area definitions
CREATE TABLE IF NOT EXISTS area_definitions (
    id SERIAL PRIMARY KEY,
    areas JSONB NOT NULL,
    image_path TEXT,
    area_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing counting results
CREATE TABLE IF NOT EXISTS counting_results (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    empty_count INTEGER NOT NULL DEFAULT 0,
    occupied_count INTEGER NOT NULL DEFAULT 0,
    total_areas INTEGER NOT NULL DEFAULT 0,
    estimated_pallets INTEGER DEFAULT 0,
    estimated_sacks INTEGER DEFAULT 0,
    estimated_weight_tons DECIMAL(10,2) DEFAULT 0,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for monitoring sessions
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    rtsp_url TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    frame_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for system settings
CREATE TABLE IF NOT EXISTS system_settings (
    id SERIAL PRIMARY KEY,
    setting_key TEXT UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_counting_results_timestamp ON counting_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_counting_results_session_id ON counting_results(session_id);
CREATE INDEX IF NOT EXISTS idx_area_definitions_created_at ON area_definitions(created_at);
CREATE INDEX IF NOT EXISTS idx_monitoring_sessions_session_id ON monitoring_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_monitoring_sessions_started_at ON monitoring_sessions(started_at);

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
('empty_threshold', '0.15', 'Threshold for determining if an area is empty'),
('pallet_height', '4', 'Number of pallet layers'),
('pallet_capacity', '20', 'Number of sacks per pallet'),
('sack_weight', '50', 'Weight per sack in kilograms'),
('frame_rate', '30', 'Processing frame rate for monitoring')
ON CONFLICT (setting_key) DO NOTHING;

-- Create a view for analytics
CREATE OR REPLACE VIEW analytics_summary AS
SELECT 
    DATE(timestamp) as date,
    AVG(occupied_count) as avg_occupied,
    AVG(empty_count) as avg_empty,
    AVG(total_areas) as avg_total_areas,
    MAX(occupied_count) as max_occupied,
    MIN(empty_count) as min_empty,
    COUNT(*) as record_count
FROM counting_results 
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Create a function to clean old data
CREATE OR REPLACE FUNCTION clean_old_counting_data(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM counting_results 
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE area_definitions IS 'Stores defined pallet areas for stock counting';
COMMENT ON TABLE counting_results IS 'Stores real-time stock counting results';
COMMENT ON TABLE monitoring_sessions IS 'Tracks monitoring sessions and their status';
COMMENT ON TABLE system_settings IS 'Stores configurable system parameters';
COMMENT ON VIEW analytics_summary IS 'Daily analytics summary for reporting';
COMMENT ON FUNCTION clean_old_counting_data IS 'Function to clean old counting data beyond specified days';
