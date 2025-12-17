"""
Speech Training Status Service
Handles training status tracking and provides foundation for real-time updates.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SpeechTrainingStatusService:
    """
    Service for handling real-time speech operations and training status updates.
    Provides foundation for WebSocket/SSE integration.
    """

    def __init__(self):
        self.active_training_sessions: Dict[str, Dict[str, Any]] = {}
        self.training_subscribers: Dict[str, List] = {}  # For future WebSocket/SSE subscribers

    async def start_training_session(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new training session and track its progress.

        Args:
            session_id: Unique identifier for the training session
            config: Training configuration

        Returns:
            Session information
        """
        session = {
            'id': session_id,
            'config': config,
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing training session...',
            'logs': [],
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }

        self.active_training_sessions[session_id] = session

        # Log the training start
        await self.log_training_event(session_id, 'info', f'Training session {session_id} started')

        return session

    async def update_training_progress(self, session_id: str, progress: float, message: str) -> None:
        """
        Update training progress for a session.

        Args:
            session_id: Training session ID
            progress: Progress percentage (0-100)
            message: Current status message
        """
        if session_id not in self.active_training_sessions:
            logger.warning(f"Attempted to update progress for unknown session: {session_id}")
            return

        session = self.active_training_sessions[session_id]
        session['progress'] = progress
        session['message'] = message
        session['last_update'] = datetime.now().isoformat()

        # Log progress update
        await self.log_training_event(session_id, 'progress', f'Progress: {progress:.1f}% - {message}')

        # Notify subscribers (WebSocket/SSE implementation)
        await self.notify_subscribers(session_id, {
            'type': 'progress_update',
            'progress': progress,
            'message': message,
            'timestamp': session['last_update']
        })

        # Broadcast via WebSocket if available
        try:
            from main import broadcast_speech_training_update
            await broadcast_speech_training_update(session_id, {
                'progress': progress,
                'message': message,
                'status': session['status']
            })
        except ImportError:
            # WebSocket broadcasting not available
            pass

    async def complete_training_session(self, session_id: str, success: bool, final_message: str = None) -> None:
        """
        Complete a training session.

        Args:
            session_id: Training session ID
            success: Whether training completed successfully
            final_message: Final status message
        """
        if session_id not in self.active_training_sessions:
            logger.warning(f"Attempted to complete unknown session: {session_id}")
            return

        session = self.active_training_sessions[session_id]
        session['status'] = 'success' if success else 'error'
        session['progress'] = 100.0 if success else 0.0
        session['message'] = final_message or ('Training completed successfully' if success else 'Training failed')
        session['end_time'] = datetime.now().isoformat()
        session['last_update'] = datetime.now().isoformat()

        # Log completion
        event_type = 'success' if success else 'error'
        await self.log_training_event(session_id, event_type, session['message'])

        # Notify subscribers
        await self.notify_subscribers(session_id, {
            'type': 'training_completed',
            'success': success,
            'message': session['message'],
            'timestamp': session['last_update']
        })

    async def log_training_event(self, session_id: str, event_type: str, message: str) -> None:
        """
        Log a training event.

        Args:
            session_id: Training session ID
            event_type: Type of event (info, warning, error, progress, success)
            message: Event message
        """
        if session_id not in self.active_training_sessions:
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'session_id': session_id
        }

        session = self.active_training_sessions[session_id]
        session['logs'].append(log_entry)

        # Keep only last 100 logs to prevent memory issues
        if len(session['logs']) > 100:
            session['logs'] = session['logs'][-100:]

        logger.info(f"Training {session_id} [{event_type}]: {message}")

    async def get_training_status(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get training status for a session or all sessions.

        Args:
            session_id: Specific session ID, or None for summary

        Returns:
            Training status information
        """
        if session_id:
            session = self.active_training_sessions.get(session_id)
            if not session:
                return {'status': 'not_found', 'message': f'Session {session_id} not found'}

            return {
                'status': session['status'],
                'progress': session['progress'],
                'message': session['message'],
                'logs': session['logs'][-10:],  # Return last 10 logs
                'start_time': session['start_time'],
                'last_update': session['last_update']
            }
        else:
            # Return summary of all active sessions
            active_sessions = {
                sid: {
                    'status': session['status'],
                    'progress': session['progress'],
                    'message': session['message'],
                    'start_time': session['start_time'],
                    'last_update': session['last_update']
                }
                for sid, session in self.active_training_sessions.items()
                if session['status'] in ['starting', 'running']
            }

            return {
                'active_sessions': active_sessions,
                'total_active': len(active_sessions)
            }

    async def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed training sessions.

        Args:
            max_age_hours: Maximum age in hours for completed sessions

        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        to_remove = []

        for session_id, session in self.active_training_sessions.items():
            if session['status'] in ['success', 'error']:
                end_time = session.get('end_time')
                if end_time:
                    end_datetime = datetime.fromisoformat(end_time)
                    age_hours = (current_time - end_datetime).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_remove.append(session_id)

        for session_id in to_remove:
            del self.active_training_sessions[session_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old training sessions")

        return len(to_remove)

    async def notify_subscribers(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Notify subscribers of training updates (for future WebSocket/SSE implementation).

        Args:
            session_id: Training session ID
            data: Update data to send to subscribers
        """
        subscribers = self.training_subscribers.get(session_id, [])
        for subscriber in subscribers:
            try:
                # This would be implemented based on WebSocket/SSE framework used
                # For now, just log the notification
                logger.debug(f"Would notify subscriber for session {session_id}: {data}")
            except Exception as e:
                logger.error(f"Failed to notify subscriber: {e}")

    # Speech-specific operations can be added here
    async def get_speech_model_status(self) -> Dict[str, Any]:
        """
        Get current speech model status.

        Returns:
            Speech model status information
        """
        # This would integrate with the existing speech service
        return {
            'current_model': 'whisper',
            'available_models': ['en', 'it', 'multi'],
            'last_updated': datetime.now().isoformat()
        }

    async def get_training_data_stats(self) -> Dict[str, Any]:
        """
        Get training data statistics.

        Returns:
            Training data statistics
        """
        # This would integrate with the existing speech training data management
        return {
            'total_audio_files': 0,
            'annotated_files': 0,
            'training_sessions': len(self.active_training_sessions),
            'language_counts': {},
            'last_updated': datetime.now().isoformat()
        }

    async def get_whisper_training_status_details(self) -> Dict[str, Any]:
        """
        Get detailed Whisper training status (for backward compatibility).

        Returns:
            Detailed training status in the format expected by the frontend
        """
        # Get the most recent active training session
        active_sessions = [
            (sid, session) for sid, session in self.active_training_sessions.items()
            if session['status'] in ['starting', 'running']
        ]

        if not active_sessions:
            return {
                'status': 'idle',
                'progress': 0,
                'message': 'No active training sessions',
                'logs': []
            }

        # Use the most recent session
        _, session = max(active_sessions, key=lambda x: x[1]['start_time'])

        return {
            'status': session['status'],
            'progress': session['progress'],
            'message': session['message'],
            'logs': session['logs'][-20:]  # Return last 20 logs for more detail
        }


# Global service instance
speech_training_status_service = SpeechTrainingStatusService()


async def get_speech_training_status(session_id: str = None) -> Dict[str, Any]:
    """
    Get speech training status - wrapper for the service.

    Args:
        session_id: Optional specific session ID

    Returns:
        Training status
    """
    return await speech_training_status_service.get_training_status(session_id)


async def get_whisper_training_status_details() -> Dict[str, Any]:
    """
    Get detailed Whisper training status - wrapper for backward compatibility.

    Returns:
        Detailed training status
    """
    return await speech_training_status_service.get_whisper_training_status_details()


async def update_speech_training_progress(session_id: str, progress: float, message: str) -> None:
    """
    Update speech training progress - wrapper for the service.

    Args:
        session_id: Training session ID
        progress: Progress percentage
        message: Status message
    """
    await speech_training_status_service.update_training_progress(session_id, progress, message)


async def complete_speech_training(session_id: str, success: bool, message: str = None) -> None:
    """
    Complete speech training session - wrapper for the service.

    Args:
        session_id: Training session ID
        success: Whether training succeeded
        message: Final message
    """
    await speech_training_status_service.complete_training_session(session_id, success, message)
