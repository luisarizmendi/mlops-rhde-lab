
import React, { useState, useEffect } from 'react';
import { DeviceService } from '../services/api';
import { Link } from 'react-router-dom';
import {
    CheckCircleIcon,
    AlertCircleIcon,
    XCircleIcon
} from 'lucide-react';

const DeviceStatusIcon = ({ isActive, hasAlarm }) => {
    if (!isActive) {
        return <XCircleIcon color="red" title="Device Offline" />;
    }

    if (hasAlarm) {
        return <AlertCircleIcon color="orange" title="Alarm Active" />;
    }

    return <CheckCircleIcon color="green" title="Device Active" />;
};

const DeviceList = () => {
    const [devices, setDevices] = useState([]);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        const fetchDevices = async () => {
            try {
                const deviceList = await DeviceService.getDevices();
                console.log(deviceList);
                setDevices(deviceList);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching devices:', error);
                setLoading(false);
            }
        };

        fetchDevices();
        const intervalId = setInterval(fetchDevices, 5005); // Refresh every 5 seconds

        return () => clearInterval(intervalId);
    }, []);

    if (loading) {
        return <div>Loading devices...</div>;
    }

    return (
        <div className="p-4">
        <h1 className="text-2xl font-bold mb-4">Devices</h1>
        <table className="w-full border-collapse">
        <thead>
        <tr className="bg-gray-200">
        <th className="p-2 text-left">Status</th>
        <th className="p-2 text-left">Device Name</th>
        <th className="p-2 text-left">Last Alive</th>
        </tr>
        </thead>
        <tbody>
        {devices.map(device => (
            <tr key={device.uuid} className="border-b">
            <td className="p-2">
            <DeviceStatusIcon
            isActive={device.is_active}
            hasAlarm={device.current_alarm_status}
            />
            </td>
            <td className="p-2">
            <Link
            to={`/device/${device.uuid}`}
            className="text-blue-600 hover:underline"
            >
            {device.name}
            </Link>
            </td>
            <td className="p-2">
            {device.last_alive_time
                ? new Date(device.last_alive_time).toLocaleString()
                : 'Never'}
                </td>
                </tr>
        ))}
        </tbody>
        </table>
        </div>
    );
};

export default DeviceList;
