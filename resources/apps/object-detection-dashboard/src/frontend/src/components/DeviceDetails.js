import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { DeviceService } from '../services/api';
import {
    AlertCircleIcon,
    CheckCircleIcon
} from 'lucide-react';

const DeviceDetails = () => {
    const { uuid } = useParams();
    const navigate = useNavigate();
    const [device, setDevice] = useState(null);
    const [deviceName, setDeviceName] = useState('');
    const [isEditing, setIsEditing] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchDeviceDetails = async () => {
            try {
                const details = await DeviceService.getDeviceDetails(uuid);
                setDevice(details);
                setDeviceName(details.name);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching device details:', error);
                setLoading(false);
            }
        };

        fetchDeviceDetails();
    }, [uuid]);

    const handleNameUpdate = async () => {
        try {
            // Update device name
            await DeviceService.updateDeviceName(uuid, deviceName);

            // Re-fetch device details to update alarms and other information
            const updatedDevice = await DeviceService.getDeviceDetails(uuid);
            setDevice(updatedDevice);  // Update device state with latest data

            setIsEditing(false);
        } catch (error) {
            console.error('Error updating device name:', error);
        }
    };

    const handleDelete = async () => {
        if (window.confirm('Are you sure you want to delete this device?')) {
            try {
                await DeviceService.deleteDevice(uuid);
                navigate('/');  // Redirect to device list
            } catch (error) {
                console.error('Error deleting device:', error);
            }
        }
    };

    if (loading) {
        return <div>Loading device details...</div>;
    }

    if (!device) {
        return <div>Device not found</div>;
    }

    // Calculate current alarm status and duration
    const currentAlarmStatus = device.alarm_logs.length > 0
    ? device.alarm_logs[0].is_alarm_on
    : false;

    const latestAlarmLog = device.alarm_logs.find(log => log.is_alarm_on);
    const alarmStartTime = latestAlarmLog
    ? new Date(latestAlarmLog.timestamp)
    : null;

    // Format the device creation and last alive times
    const deviceCreatedTime = device.created_at ? new Date(device.created_at) : null;
    const lastAliveTime = device.last_alive_time ? new Date(device.last_alive_time) : null;

    return (
        <div className="p-4">
        {/* Return to Main Screen Button */}
        <div className="mb-4">
        <button
        onClick={() => navigate('/')}
        className="bg-gray-300 px-4 py-2 rounded"
        >
        Return to Main Screen
        </button>
        </div>

        {/* Device Details Header */}
        <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
        {isEditing ? (
            <div className="flex items-center">
            <input
            type="text"
            value={deviceName}
            onChange={(e) => setDeviceName(e.target.value)}
            className="border p-1 mr-2"
            />
            <button
            onClick={handleNameUpdate}
            className="bg-blue-500 text-white px-2 py-1 rounded mr-2"
            >
            Save
            </button>
            <button
            onClick={() => {
                setDeviceName(device.name);
                setIsEditing(false);
            }}
            className="bg-gray-300 px-2 py-1 rounded"
            >
            Cancel
            </button>
            </div>
        ) : (
            <>
            <h1 className="text-2xl font-bold mr-4">
            {device.name || 'Unnamed Device'}
            </h1>
            <button
            onClick={() => setIsEditing(true)}
            className="bg-gray-300 px-2 py-1 rounded"
            >
            Edit
            </button>
            </>
        )}
        </div>
        <button
        onClick={handleDelete}
        className="bg-red-500 text-white px-2 py-1 rounded"
        >
        Delete
        </button>
        </div>

        {/* Alarm Status */}
        <div>
        <p>
        Current Alarm Status:{' '}
        {currentAlarmStatus ? (
            <span className="text-red-500">
            <AlertCircleIcon /> Active
            </span>
        ) : (
            <span className="text-green-500">
            <CheckCircleIcon /> Inactive
            </span>
        )}
        </p>
        {alarmStartTime && (
            <p>
            Alarm active since: {alarmStartTime.toLocaleString()}
            </p>
        )}
        </div>

        {/* Device Created and Last Alive Times */}
        <div className="mt-4">
        {deviceCreatedTime && (
            <p>
            Device Created At: {deviceCreatedTime.toLocaleString()}
            </p>
        )}
        {lastAliveTime && (
            <p>
            Last Alive Signal Received: {lastAliveTime.toLocaleString()}
            </p>
        )}
        </div>

        {/* Alarm Log List */}
        <div className="mt-4">
        <h2 className="text-xl font-bold">Alarm History</h2>
        <ul>
        {device.alarm_logs.map((log, index) => (
            <li key={index} className="border-b py-2">
            <p>{log.message}</p>
            <p>
            Status: {log.is_alarm_on ? 'Active' : 'Inactive'}
            </p>
            <p>
            Time: {new Date(log.timestamp).toLocaleString()}
            </p>
            </li>
        ))}
        </ul>
        </div>
        </div>
    );
};

export default DeviceDetails;
