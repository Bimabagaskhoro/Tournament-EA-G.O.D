import queue
import threading
from uuid import UUID
from typing import Optional

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob
from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from core.models.utility_models import TextJob
from miner.logic.job_handler import start_tuning_container
from miner.logic.job_handler import start_tuning_container_diffusion

# Import new orchestrator
try:
    from subtext.training_orchestrator import TrainingOrchestrator, JobPriority
    from subtext.smart_training_handler import execute_job_with_smart_allocation
    ORCHESTRATOR_AVAILABLE = True
    logger = get_logger(__name__)
    logger.info("Training Orchestrator available - parallel training enabled")
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning(f"Training Orchestrator not available - fallback to sequential: {e}")


class TrainingWorker:
    def __init__(self):
        logger.info("=" * 80)
        logger.info("STARTING A TRAINING WORKER")
        logger.info("=" * 80)

        # Choose between orchestrator and legacy mode
        if ORCHESTRATOR_AVAILABLE:
            # Use the new parallel orchestrator
            self.orchestrator = TrainingOrchestrator(
                training_function=self._execute_training_job
            )
            self.job_store: dict[str, Job] = {}
            self.legacy_mode = False
            logger.info("Training Worker initialized with parallel orchestrator")
        else:
            # Fallback to legacy sequential mode
            self.orchestrator = None
            self.job_queue: queue.Queue[Job] = queue.Queue()
            self.job_store: dict[str, Job] = {}
            self.thread = threading.Thread(target=self._legacy_worker, daemon=True)
            self.thread.start()
            self.legacy_mode = True
            logger.info("Training Worker initialized in legacy sequential mode")
        
        self.docker_client = docker.from_env()

    def _execute_training_job(self, job: Job):
        """
        Execute a training job - used by the orchestrator.
        This function handles the actual training execution.
        """
        try:
            if isinstance(job, TextJob):
                # Use smart training handler if available, otherwise fallback
                if ORCHESTRATOR_AVAILABLE:
                    execute_job_with_smart_allocation(job)
                else:
                    start_tuning_container(job)
            elif isinstance(job, DiffusionJob):
                start_tuning_container_diffusion(job)
            else:
                raise ValueError(f"Unsupported job type: {type(job)}")
                
        except Exception as e:
            logger.error(f"Error executing training job {job.job_id}: {str(e)}")
            job.error_message = str(e)
            raise

    def _legacy_worker(self):
        """Legacy sequential worker - fallback when orchestrator is not available"""
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            try:
                self._execute_training_job(job)
                job.status = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                self.job_queue.task_done()

    def enqueue_job(self, job: Job, priority: Optional[str] = None):
        """
        Enqueue a job for training.
        
        Args:
            job: Job to be scheduled
            priority: Job priority ("high", "normal", "low") - only used in orchestrator mode
        """
        self.job_store[job.job_id] = job
        
        if self.legacy_mode:
            # Legacy sequential mode
            self.job_queue.put(job)
            logger.info(f"Enqueued job {job.job_id} in legacy sequential mode")
        else:
            # Orchestrator parallel mode
            priority_map = {
                "high": JobPriority.HIGH,
                "normal": JobPriority.NORMAL,
                "low": JobPriority.LOW
            }
            job_priority = priority_map.get(priority, JobPriority.NORMAL)
            
            success = self.orchestrator.enqueue_job(job, job_priority)
            if success:
                logger.info(f"Enqueued job {job.job_id} with priority {job_priority.name} in orchestrator")
            else:
                logger.error(f"Failed to enqueue job {job.job_id} in orchestrator")

    def get_status(self, job_id: UUID) -> JobStatus:
        """Get the status of a job"""
        if self.legacy_mode:
            job = self.job_store.get(str(job_id))
            return job.status if job else JobStatus.NOT_FOUND
        else:
            return self.orchestrator.get_job_status(str(job_id))

    def get_job_info(self, job_id: str) -> Optional[dict]:
        """Get detailed job information (only available in orchestrator mode)"""
        if not self.legacy_mode and self.orchestrator:
            return self.orchestrator.get_job_info(job_id)
        else:
            # Return basic info in legacy mode
            job = self.job_store.get(job_id)
            if job:
                return {
                    "job_id": job_id,
                    "status": job.status.value,
                    "job_type": "text" if isinstance(job, TextJob) else "diffusion",
                    "error_message": getattr(job, 'error_message', None),
                    "legacy_mode": True
                }
            return None

    def get_training_stats(self) -> dict:
        """Get comprehensive training statistics"""
        if not self.legacy_mode and self.orchestrator:
            return self.orchestrator.get_orchestrator_stats()
        else:
            # Return basic stats in legacy mode
            total_jobs = len(self.job_store)
            status_counts = {}
            for job in self.job_store.values():
                status = job.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "legacy_mode": True,
                "total_jobs": total_jobs,
                "status_breakdown": status_counts,
                "queue_size": self.job_queue.qsize() if hasattr(self, 'job_queue') else 0
            }

    def list_waiting_jobs(self) -> list:
        """List jobs waiting to be scheduled (only available in orchestrator mode)"""
        if not self.legacy_mode and self.orchestrator:
            return self.orchestrator.list_waiting_jobs()
        else:
            return []

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        if not self.legacy_mode and self.orchestrator:
            return self.orchestrator.cancel_job(job_id)
        else:
            # In legacy mode, we can only cancel if it's not running
            job = self.job_store.get(job_id)
            if job and job.status in [JobStatus.PENDING, JobStatus.NOT_FOUND]:
                job.status = JobStatus.FAILED
                job.error_message = "Cancelled by user"
                return True
            return False

    def shutdown(self):
        """Shutdown the training worker"""
        logger.info("Shutting down Training Worker...")
        
        if self.legacy_mode:
            # Legacy mode shutdown
            self.job_queue.put(None)  # Signal shutdown
            if hasattr(self, 'thread'):
                self.thread.join()
        else:
            # Orchestrator mode shutdown
            if self.orchestrator:
                self.orchestrator.shutdown()
        
        # Close Docker client
        if hasattr(self, 'docker_client'):
            self.docker_client.close()
        
        logger.info("Training Worker shutdown complete")
